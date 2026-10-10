// ============================================================================
// overlay.cpp —— 极速透明叠加层（无边框 + 鼠标穿透 + 置顶），pybind11 模块
//
// 编译（MSVC + Windows SDK + pybind11）：
//   pip install pybind11
//   cl /O2 /std:c++17 /EHsc /LD overlay.cpp ^
//      /I"%PYTHON_INCLUDE%" ^
//      /link /LIBPATH:"%PYTHON_LIB%" ^
//      user32.lib gdi32.lib dwmapi.lib winmm.lib ^
//      /OUT:overlay.pyd
//
// 用法：
//   import overlay
//   ov = overlay.Overlay()
//   ov.show()
//   ov.clear()
//   ov.draw_box(100, 100, 300, 400, 0xFFFF0000, 2)
//   ov.present()          # 一帧提交
//   ov.hide()
//
// ----------------------------------------------------------------------------
// 一、为什么这套方案能"极速"，以及它的关键约束
// ----------------------------------------------------------------------------
// 普通窗口要经历 WM_PAINT -> BeginPaint -> GDI 绘制 -> DWM 重定向 的完整流程，
// 每一帧都要走消息队列，帧率被消息循环卡住。
//
// 分层窗口（WS_EX_LAYERED）+ UpdateLayeredWindow() 走的是另一条路：
// 我们把一整块 32bpp 位图直接交给 DWM 合成器，DWM 拿它作为窗口的“最终内容”，
// 完全不经过 WM_PAINT，也不需要消息循环。因此：
//   * 本文件里没有 WndProc 自定义逻辑，lpfnWndProc 直接用 DefWindowProcW；
//   * 绘制函数（draw_box/draw_line/...）只是往一块 uint32_t 数组里写内存，
//     零 GDI 调用，零系统调用；只有 present() 才真正触发一次内核调用。
//
// 由此产生两条“必须遵守”的约束，也是本项目原来出问题的根源：
//   1) 同一个分层窗口不能混用 SetLayeredWindowAttributes() 和
//      UpdateLayeredWindow()。前者是“整窗统一 alpha / 颜色键”模式，后者是
//      “逐像素 alpha”模式，两者互斥（MSDN 明确说明不支持并发使用）。
//      混用会让窗口停留在整窗 alpha 语义下，导致逐像素 alpha 被忽略 ——
//      表现就是“窗口创建了但内容画不出来”，并在 ShowWindow 之后闪一帧纯黑全屏。
//      所以本实现只使用 UpdateLayeredWindow，绝不调用 SetLayeredWindowAttributes。
//   2) UpdateLayeredWindow 传 AC_SRC_ALPHA 时，源位图必须是**预乘 alpha**
//      (premultiplied) 的 32bpp BGRA。见下面“二、像素格式”。
//
// 另外一个和“游戏窗口”相关的现实限制：只有当游戏处于无边框全屏/窗口化时，
// DWM 才会把本叠加层合成到游戏之上；独占全屏 (exclusive fullscreen) 直接写
// 显卡扫描输出，绕过 DWM，任何 Win32 叠加层都无法覆盖。
//
// ----------------------------------------------------------------------------
// 二、像素格式（本文件最容易踩坑的地方）
// ----------------------------------------------------------------------------
// Python 侧传进来的颜色约定为 0xAARRGGBB（一个无符号 32 位整数）。
// 例如 0xFF00F000 = 不透明的绿色 (R=0x00, G=0xF0, B=0x00)。
//
// 而 x86 是小端序，一个 uint32_t 在内存中的字节顺序是 低字节 -> 高字节，
// 即 B, G, R, A。这恰好就是 Windows 32bpp DIB 要求的 BGRA 布局，
// 所以可以直接把 uint32_t 数组当作位图像素缓冲，无需逐通道拷贝：
//
//   bits_[i] = 0xAARRGGBB
//   内存: [i*4+0]=BB  [i*4+1]=GG  [i*4+2]=RR  [i*4+3]=AA
//
// 但 bits_ 中存放的不是原始颜色，而是**预乘后**的颜色：
//   stored_R = R * A / 255   （G、B 同理）
// 这样 present() 交给 DWM 的数据才符合 AC_SRC_ALPHA 的语义。
// 所有写入像素的入口（put/blend/draw_text 的补偿遍历）都负责把数据转成预乘格式，
// 因此 buffer 内部格式始终统一，clear() 写 0 也就自然表示“完全透明”。
//
// 坐标：(0,0) 在左上角。CreateDIBSection 用 biHeight = -height 声明自上而下
// (top-down) 的行序，这样 bits_[y*w+x] 的索引顺序才和屏幕坐标一致；
// 如果用默认的 bottom-up，图像会上下颠倒。
// ============================================================================

#include <pybind11/pybind11.h>

#include <windows.h>
#include <dwmapi.h>
#include <mmsystem.h>

#include <cstdint>
#include <cstdlib>   // std::abs
#include <cstring>   // std::memset
#include <stdexcept> // std::runtime_error
#include <string>    // std::to_string / std::string

// 用 #pragma comment 让链接器自动带上这些库，避免调用 cl 时漏写依赖。
#pragma comment(lib, "user32.lib")
#pragma comment(lib, "gdi32.lib")
#pragma comment(lib, "dwmapi.lib")
#pragma comment(lib, "winmm.lib")

namespace py = pybind11;

// ---------------------------------------------------------------------------
// 颜色/alpha 的小工具。全部 static inline：只是给编译器的提示性内联，
// 且把符号限制在本翻译单元内，不会和其他 .cpp 里的同名函数打架。
// ---------------------------------------------------------------------------

// 从 0xAARRGGBB 中取单个通道
static inline uint32_t alpha_of(uint32_t argb) { return (argb >> 24) & 0xFFu; }
static inline uint32_t red_of(uint32_t argb)   { return (argb >> 16) & 0xFFu; }
static inline uint32_t green_of(uint32_t argb) { return (argb >> 8)  & 0xFFu; }
static inline uint32_t blue_of(uint32_t argb)  { return argb & 0xFFu; }

// 把 (v * a) 除回 8 位范围。/255 是整数除法很慢，这里用 (x + 127) / 255 的
// 等价式（编译器会优化成移位+加法），同时得到四舍五入而不是向下取整，
// 避免半透明色通道被截断成 0 而出现“边缘发黑”。
static inline uint32_t mul_div255(uint32_t v, uint32_t a) {
    return (v * a + 127u) / 255u;
}

// 非预乘 0xAARRGGBB -> 预乘 BGRA（即本 buffer 的存储格式）
static inline uint32_t to_premultiplied(uint32_t argb) {
    uint32_t a = alpha_of(argb);
    if (a == 255u) return argb;          // 快路径：不透明时预乘等于原值
    if (a == 0u)   return 0u;            // 完全透明 -> 全 0（同时清掉颜色，避免脏 RGB）
    uint32_t r = mul_div255(red_of(argb),   a);
    uint32_t g = mul_div255(green_of(argb), a);
    uint32_t b = mul_div255(blue_of(argb),  a);
    return (a << 24) | (r << 16) | (g << 8) | b;
}

// ============================================================================
// Overlay —— 一个全屏分层窗口 + 一块 CPU 像素缓冲
//
// 生命周期：show() -> [clear/draw_*/present]* -> hide()
// 所有绘制方法在 show() 之前调用是安全的（bits_ 为空时直接返回，不会崩）。
// ============================================================================
class Overlay {
public:
    Overlay() = default;

    // 析构时兜底清理。注意：这里不能抛异常（Python GC 可能在解释器关闭阶段
    // 触发析构），所以 hide() 内部所有失败都被吞掉。
    ~Overlay() {
        try { hide(); } catch (...) {}
    }

    // ------------------------------------------------------------------
    // show(): 创建窗口 + 分配位图，进入可绘制状态
    // 返回 true 表示成功；失败时抛 std::runtime_error（Python 侧看到
    // RuntimeError），这样“窗口没画出来”这类问题不再静默失败。
    // ------------------------------------------------------------------
    bool show() {
        if (hwnd_) return true;   // 幂等：已经显示过就直接返回

        // ---- 0) DPI 感知 ----
        // 必须在创建任何窗口之前设置。Python 的 python.exe 默认是 DPI-unaware：
        // 在 125%/150% 缩放的屏幕上，GetSystemMetrics 会返回“逻辑像素”，
        // 于是窗口只有屏幕的一部分或被系统位图拉伸（发虚），
        // 更严重的是：屏幕捕获（dxcam）拿到的永远是物理像素，
        // 两套坐标系错位会让 YOLO 框整体偏移。
        // SetProcessDpiAwarenessContext 只有 Win10 1703+ 才有，且不同 SDK 版本
        // 的头文件里可能没有对应的类型声明，所以用 GetProcAddress 动态取函数、
        // 并把不透明的 DPI_AWARENESS_CONTEXT 句柄当作 void* 传递（ABI 相同）。
        // -4 == DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
        using SetDpiCtxFn = BOOL(WINAPI*)(void*);
        HMODULE user32 = GetModuleHandleW(L"user32.dll");
        auto set_dpi = reinterpret_cast<SetDpiCtxFn>(
            GetProcAddress(user32, "SetProcessDpiAwarenessContext"));
        if (set_dpi) {
            set_dpi(reinterpret_cast<void*>(-4));
        } else {
            // 老系统退化成 System DPI Aware（Vista+ 一直有），至少不会被拉伸。
            SetProcessDPIAware();
        }

        // ---- 1) 屏幕尺寸 ----
        // SM_CXSCREEN/SM_CYSCREEN = 主显示器物理分辨率（在 DPI 感知已生效时）。
        // 本叠加层只覆盖主屏；CS2 无边框全屏通常就在主屏。
        screen_w_ = GetSystemMetrics(SM_CXSCREEN);
        screen_h_ = GetSystemMetrics(SM_CYSCREEN);
        if (screen_w_ <= 0 || screen_h_ <= 0) {
            throw std::runtime_error("show: GetSystemMetrics 返回了非法屏幕尺寸");
        }

        // ---- 2) 注册窗口类（进程内只需一次）----
        WNDCLASSEXW wc = {};
        wc.cbSize        = sizeof(wc);
        // 分层窗口不需要 WM_PAINT 处理， DefWindowProcW 足以应付
        // WM_CREATE/WM_DESTROY 等默认消息。
        wc.lpfnWndProc   = DefWindowProcW;
        wc.hInstance     = GetModuleHandleW(nullptr);
        wc.hCursor       = LoadCursor(nullptr, IDC_ARROW);
        // hbrBackground 保持 nullptr（= 不填背景）。
        // 若设成某个画刷，系统在 WM_ERASEBKGND 时会填色，
        // 对分层窗口毫无意义，还会浪费一次全屏填充。
        wc.hbrBackground = nullptr;
        wc.lpszClassName = kClassName;

        if (!RegisterClassExW(&wc)) {
            // 1418 == ERROR_CLASS_DOES_NOT_EXIST；若已经注册过（重复 show/hide）
            // 则报 1414/ERROR_CLASS_ALREADY_EXISTS，那种情况可以继续。
            if (::GetLastError() != ERROR_CLASS_ALREADY_EXISTS) {
                throw std::runtime_error("show: RegisterClassExW 失败, GetLastError="
                                         + std::to_string(::GetLastError()));
            }
        }

        // ---- 3) 创建窗口 ----
        // 扩展样式逐条说明：
        //   WS_EX_LAYERED     分层窗口：允许逐像素 alpha，是 UpdateLayeredWindow 的前提
        //   WS_EX_TRANSPARENT 鼠标穿透：命中测试直接跳过本窗口，落到下面的窗口
        //                    （必须与 WS_EX_LAYERED 同时使用才可靠）
        //   WS_EX_TOPMOST     置顶：Z 序位于所有非置顶窗口之上
        //   WS_EX_NOACTIVATE  永不获得键盘焦点：不影响玩家操作/输入
        // 客户区样式：
        //   WS_POPUP          无标题栏、无边框、无系统菜单，客户区 = 整个窗口
        // 这里先不 ShowWindow：等首帧 present() 之后再显示，
        // 避免“窗口已出现但 buffer 还没内容”造成的黑屏闪烁。
        hwnd_ = CreateWindowExW(
            WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST | WS_EX_NOACTIVATE,
            kClassName, L"",
            WS_POPUP,
            0, 0, screen_w_, screen_h_,
            nullptr, nullptr, GetModuleHandleW(nullptr), nullptr);
        if (!hwnd_) {
            hwnd_ = nullptr;
            throw std::runtime_error("show: CreateWindowExW 失败, GetLastError="
                                     + std::to_string(::GetLastError()));
        }

        // 关闭该窗口的 DWM 过渡动画，减少合成延迟（失败可忽略）。
        BOOL disable_transitions = TRUE;
        DwmSetWindowAttribute(hwnd_, DWMWA_TRANSITIONS_FORCEDISABLED,
                              &disable_transitions, sizeof(disable_transitions));

        // ---- 4) 准备 DC 层级 ----
        // DC（设备上下文）本质上是“某块位图 + 一套绘图状态”的句柄。这里有三个：
        //   screen_dc_ : 屏幕 DC，仅作为 present() 的 hdcDst（“目标显示器”）
        //                以及 CreateCompatibleDC 的格式参考。
        //   mem_dc_    : 与屏幕兼容的内存 DC，是个“空画布”，必须 SelectObject
        //                一张位图进去才能画图；它就是 present() 的 hdcSrc。
        //   hbitmap_   : 真正存放像素的对象，其像素内存由系统映射，
        //                地址通过 CreateDIBSection 的 ppvBits 出参还给我们 ->
        //                这就是 bits_ 这个裸指针的来源。
        screen_dc_ = GetDC(nullptr);
        if (!screen_dc_) {
            cleanup_gdi();
            DestroyWindow(hwnd_); hwnd_ = nullptr;
            throw std::runtime_error("show: GetDC(nullptr) 失败");
        }
        mem_dc_ = CreateCompatibleDC(screen_dc_);
        if (!mem_dc_) {
            cleanup_gdi();
            DestroyWindow(hwnd_); hwnd_ = nullptr;
            throw std::runtime_error("show: CreateCompatibleDC 失败");
        }

        // ---- 5) 创建与 bits_ 共享内存的 DIB Section ----
        BITMAPINFO bmi = {};
        bmi.bmiHeader.biSize        = sizeof(BITMAPINFOHEADER);
        bmi.bmiHeader.biWidth       = screen_w_;
        // 负高度 = top-down（第一行是屏幕最上面一行）。
        // 见文件头“二、像素格式”：这样 bits_[y*w+x] 才与屏幕坐标同向。
        bmi.bmiHeader.biHeight      = -screen_h_;
        bmi.bmiHeader.biPlanes      = 1;      // 必须为 1，历史遗留字段
        bmi.bmiHeader.biBitCount    = 32;     // 每像素 32 位 -> BGRA
        bmi.bmiHeader.biCompression = BI_RGB; // 不压缩；alpha 放在第 4 字节

        void* bits = nullptr;
        hbitmap_ = CreateDIBSection(mem_dc_, &bmi, DIB_RGB_COLORS, &bits, nullptr, 0);
        if (!hbitmap_ || !bits) {
            cleanup_gdi();
            DestroyWindow(hwnd_); hwnd_ = nullptr;
            throw std::runtime_error("show: CreateDIBSection 失败, GetLastError="
                                     + std::to_string(::GetLastError()));
        }
        // bits_ 指向系统为 DIB 分配的像素内存。它的所有权属于 hbitmap_：
        // 绝不能 free/delete，只能靠 DeleteObject(hbitmap_) 释放。
        bits_ = static_cast<uint32_t*>(bits);

        // 把 DIB 选进内存 DC，之后 GDI（draw_text）才会写到这块内存上。
        // SelectObject 返回上一个 GDI 对象，这里 DC 是新建的，返回值必然为 nullptr，
        // 但保留写法：退出作用域前把旧对象选回去，是 GDI 的标配（避免 DC 带着
        // 已删除位图被使用）。
        old_bmp_ = SelectObject(mem_dc_, hbitmap_);

        // ---- 6) 初始化为全透明 ----
        clear();

        // ---- 7) 提高系统定时器分辨率到 1ms ----
        // 只影响 Sleep() 的粒度；上层若需要稳定帧间隔做节流会用到。
        timeBeginPeriod(1);
        timer_started_ = true;

        // 记录创建窗口的线程。Win32 要求 DestroyWindow 必须由创建它的线程调用，
        // 而 pybind11 对象可能在不同 Python 线程上被 GC 掉。
        owner_thread_ = GetCurrentThreadId();
        return true;
    }

    // ------------------------------------------------------------------
    // hide(): 销毁窗口并释放所有 GDI 资源。可重复调用，不抛异常。
    // ------------------------------------------------------------------
    void hide() {
        if (hwnd_) {
            // 只有创建线程能安全销毁；否则只隐藏，把销毁留给下一次
            // （或进程退出时由系统回收），避免未定义行为。
            if (GetCurrentThreadId() == owner_thread_) {
                DestroyWindow(hwnd_);
            } else {
                ShowWindow(hwnd_, SW_HIDE);
            }
            hwnd_ = nullptr;
        }
        cleanup_gdi();

        if (timer_started_) {
            timeEndPeriod(1);   // 必须与 timeBeginPeriod 成对，否则系统定时器
            timer_started_ = false;  // 分辨率被永久拉低，影响整机功耗
        }
        bits_      = nullptr;
        screen_w_  = 0;
        screen_h_  = 0;
    }

    // ------------------------------------------------------------------
    // clear(): 整块 buffer 置 0。
    // 因为 0 表示 alpha=0（完全透明），一次 memset 就等于“擦干净这一帧”。
    // 4 = 每像素字节数；用 size_t 乘法避免 3840*2160*4 在 int 里溢出。
    // ------------------------------------------------------------------
    void clear() {
        if (!bits_) return;
        std::memset(bits_, 0, pixel_count() * 4u);
    }

    // ------------------------------------------------------------------
    // draw_box(): 空心矩形（检测框的主力函数），color = 0xAARRGGBB。
    // thickness 是“边框向内扩展的像素数”。
    //
    // 实现说明：原来的写法是画 thickness 圈逐渐内缩的矩形轮廓，
    // 那是 O(thickness × 周长) 的重复写同一批像素，thickness 一大就明显浪费
    // （比如 thickness=20 会把角上重画 20 次）。这里改成 4 个实心条带，
    // 每条带只覆盖属于它的像素，总写入量恰好等于矩形框的面积。
    // ------------------------------------------------------------------
    void draw_box(int x1, int y1, int x2, int y2, uint32_t color, int thickness) {
        if (!bits_) return;
        if (x1 > x2) std::swap(x1, x2);   // 允许任意方向的角点
        if (y1 > y2) std::swap(y1, y2);
        if (thickness < 1) thickness = 1;

        // x2/y2 按“闭区间”理解（含边界像素），所以宽高各 +1。
        int w = x2 - x1 + 1;
        int h = y2 - y1 + 1;
        // thickness 超过短边时，上下左右条带会互相重叠 —— 直接钳到短边，
        // 效果退化成实心块，不会出现负坐标或越界。
        if (thickness > w) thickness = w;
        if (thickness > h) thickness = h;

        // 上、下两条横带：横跨整个宽度
        band(x1, y1, x2, y1 + thickness - 1, color);
        band(x1, y2 - thickness + 1, x2, y2, color);
        // 左、右两条竖带：只画中间部分（上下带已经覆盖了端头，不重复写）
        band(x1, y1 + thickness, x1 + thickness - 1, y2 - thickness, color);
        band(x2 - thickness + 1, y1 + thickness, x2, y2 - thickness, color);
    }

    // ------------------------------------------------------------------
    // fill_box(): 实心矩形。color 的 alpha 参与真正的混合，
    // 所以 0x40FF0000 会画出 25% 不透明的红色半透明块（适合做区域高亮）。
    // ------------------------------------------------------------------
    void fill_box(int x1, int y1, int x2, int y2, uint32_t color) {
        if (!bits_) return;
        if (x1 > x2) std::swap(x1, x2);
        if (y1 > y2) std::swap(y1, y2);
        for (int y = y1; y <= y2; ++y)
            for (int x = x1; x <= x2; ++x)
                blend(x, y, color);
    }

    // ------------------------------------------------------------------
    // draw_line(): Bresenham 直线（用来画十字准星、连接线）。
    // thickness > 1 时以当前点为中心铺一个方形笔刷。
    // ------------------------------------------------------------------
    void draw_line(int x1, int y1, int x2, int y2, uint32_t color, int thickness) {
        if (!bits_) return;

        // dx/dy 取绝对值，dy 带负号；sx/sy 是 x/y 的步进方向（+1/-1）。
        // err 是“决策变量”：每走一步累加，用它决定下一步该在哪个方向前进，
        // 全程只用整数加减，不用除法 —— 这是 Bresenham 的核心。
        int dx = std::abs(x2 - x1), dy = -std::abs(y2 - y1);
        int sx = x1 < x2 ? 1 : -1;
        int sy = y1 < y2 ? 1 : -1;
        int err = dx + dy;

        const int half = thickness > 1 ? thickness / 2 : 0;
        while (true) {
            if (half == 0) {
                put(x1, y1, color);
            } else {
                for (int oy = -half; oy <= half; ++oy)
                    for (int ox = -half; ox <= half; ++ox)
                        put(x1 + ox, y1 + oy, color);
            }
            if (x1 == x2 && y1 == y2) break;   // 到达终点（必达，不会死循环）
            int e2 = 2 * err;
            if (e2 >= dy) { err += dy; x1 += sx; }
            if (e2 <= dx) { err += dx; y1 += sy; }
        }
    }

    // ------------------------------------------------------------------
    // draw_text(): 用 GDI 光栅化文字（中文/英文皆可，走系统字体）。
    // 这是唯一使用 GDI 的绘制路径，比纯内存写入慢，只用于少量标签。
    //
    // 【原版最隐蔽的 bug】GDI 往 32bpp DIB 里写字素时，只填 BGR 三个字节，
    // 第 4 个字节（rgbReserved，在我们这里就是 alpha）恒为 0。
    // 而 present() 用的是 AC_SRC_ALPHA —— alpha=0 意味着完全透明，
    // 于是文字被正确地画进了 buffer，却在合成阶段 100% 消失，
    // 看起来就像“draw_text 完全没生效”。
    //
    // 【修法】draw_text 之后对文字矩形做一次补偿遍历：
    // GDI 在纯黑背景上输出的 RGB 值本身就等价于“预乘后的颜色”
    // （抗锯齿边缘是暗化的 RGB，越接近背景越暗，这正是预乘的定义），
    // 所以把 alpha 补齐为该像素 max(r,g,b) 通道值，
    // 就得到一个既可见、又能保留抗锯齿柔和边缘的预乘像素。
    //
    // 注意：TextOutW 是直接覆盖，会抹掉该区域已有内容（包括 alpha）。
    // 一般把标签当作最上层元素来用，符合预期；如需保留背景请先调 draw_text
    // 再画其他图元。
    // ------------------------------------------------------------------
    void draw_text(int x, int y, const std::wstring& text, uint32_t color, int size) {
        if (!hwnd_ || !mem_dc_ || text.empty()) return;

        // color: 0xAARRGGBB -> GDI 的 COLORREF: 0x00BBGGRR（注意 GDI 没有 alpha）
        SetTextColor(mem_dc_, RGB(red_of(color), green_of(color), blue_of(color)));
        SetBkMode(mem_dc_, TRANSPARENT);   // 不画背景矩形，否则会留下一块黑底

        // 字体。第一个参数是字符高度（像素，正值=按字面高度取整）。
        HFONT font = CreateFontW(
            size, 0, 0, 0, FW_BOLD, FALSE, FALSE, FALSE,
            DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
            ANTIALIASED_QUALITY,           // 灰度抗锯齿（配合上面的 max 通道补 alpha）
            DEFAULT_PITCH | FF_DONTCARE, L"Microsoft YaHei");
        if (!font) return;

        // GDI 对象使用前必须 SelectObject 进 DC，用完必须换回旧的再 DeleteObject，
        // 否则句柄泄漏（GDI 对象有进程上限，泄漏几万个会整进程卡死）。
        HGDIOBJ old_font = SelectObject(mem_dc_, font);

        // 量一下尺寸，只遍历这块小区域，不要扫全屏。
        SIZE ext = {0, 0};
        GetTextExtentPoint32W(mem_dc_, text.c_str(), static_cast<int>(text.size()), &ext);

        TextOutW(mem_dc_, x, y, text.c_str(), static_cast<int>(text.size()));

        SelectObject(mem_dc_, old_font);
        DeleteObject(font);

        // ---- 补偿 alpha ----
        fixup_gdi_alpha(x, y, ext.cx, ext.cy);
    }

    // ------------------------------------------------------------------
    // present(): 把整块 buffer 一次性交给 DWM 合成 —— 一帧的结束。
    // 这是本模块唯一的“上屏”动作；在此之前所有 draw_* 只改内存，
    // 因此调用顺序天然是双缓冲（更准确说：写后台位图 + 一次提交），
    // 不会看到画到一半的撕裂帧。
    // ------------------------------------------------------------------
    void present() {
        if (!hwnd_ || !bits_) return;

        POINT src = {0, 0};        // 从 mem_dc_ 位图的左上角取
        POINT dst = {0, 0};        // 贴到屏幕（= 窗口客户区）左上角
        SIZE  sz  = {screen_w_, screen_h_};

        // BLENDFUNCTION 四个字段逐位解释：
        //   BlendOp    = AC_SRC_OVER      标准 Src-over-Dst 合成
        //   SrcBlendOp = 0                不做 Dst 透明键
        //   Alpha      = 255              整层不额外衰减（逐像素 alpha 才是主角）
        //   Source     = AC_SRC_ALPHA     使用位图自带的 per-pixel alpha
        //                —— 这一位要求源像素必须是预乘格式，见文件头。
        BLENDFUNCTION bf = {AC_SRC_OVER, 0, 255, AC_SRC_ALPHA};

        // 参数依次：窗口 / 目标屏幕 DC / 目标位置 / 尺寸 / 源内存 DC / 源位置 /
        //           透明色键(未用=0) / 混合描述 / 标志
        // 这里刻意不传 ULW_COLORKEY、也不传 ULW_OPAQUE：只要 AC_SRC_ALPHA。
        if (!UpdateLayeredWindow(hwnd_, screen_dc_, &dst, &sz, mem_dc_, &src,
                                 0, &bf, ULW_ALPHA)) {
            // 原版把返回值直接丢了，失败时静默 —— 这正是“看不出为什么没画出来”
            // 的原因。现在报错给 Python，附带 Win32 错误码。
            throw std::runtime_error("present: UpdateLayeredWindow 失败, GetLastError="
                                     + std::to_string(::GetLastError()));
        }

        // 首次 present 之后才让窗口可见：此时内容已就绪，不会闪黑屏。
        if (!shown_) {
            // 用 SetWindowPos 而不是 ShowWindow：同时重新断言“置顶 + 不激活”。
            // SWP_NOACTIVATE 保证抢不走游戏焦点。
            SetWindowPos(hwnd_, HWND_TOPMOST, 0, 0, 0, 0,
                         SWP_SHOWWINDOW | SWP_NOACTIVATE | SWP_NOMOVE | SWP_NOSIZE);
            shown_ = true;
        }
    }

    // ---- 只读信息：Python 侧用它把 YOLO 的像素坐标换算成比例坐标 ----
    int width()  const { return screen_w_; }
    int height() const { return screen_h_; }

private:
    // 一个像素块的总像素数。size_t 转换必须在乘法之前，否则 4K 屏会 int 溢出。
    size_t pixel_count() const {
        return static_cast<size_t>(screen_w_) * static_cast<size_t>(screen_h_);
    }

    // ------------------------------------------------------------------
    // put(): 覆盖写入一个像素（不做混合）。color 是外部约定的 0xAARRGGBB，
    // 这里负责转成 buffer 内部的预乘格式。alpha=0 等效于“擦除该像素”。
    //
    // 边界检查用 >= 上界，因为 x2/y2 是闭区间坐标，越界像素必须丢掉
    // ——少了这两行，画到屏幕边缘的检测框就会写出 buffer 之外（内存越界）。
    // ------------------------------------------------------------------
    inline void put(int x, int y, uint32_t argb) {
        if (x < 0 || y < 0 || x >= screen_w_ || y >= screen_h_) return;
        bits_[static_cast<size_t>(y) * static_cast<size_t>(screen_w_) + x] =
            to_premultiplied(argb);
    }

    // ------------------------------------------------------------------
    // blend(): 在**预乘空间**里做 source-over 合成。
    // 预乘格式下的 over 算式特别简洁（一次乘加即可，无需除法归一化）：
    //     out = src + dst * (1 - srcA)
    // 对 R/G/B/A 四个通道一视同仁，这正是选用预乘格式的动机。
    //
    // 原版的错误：把结果 alpha 硬写成 0xFF，于是任何 alpha<255 的填充
    // 都会变成完全不透明，且完全忽略了 dst 已有的 alpha。
    // ------------------------------------------------------------------
    inline void blend(int x, int y, uint32_t argb) {
        if (x < 0 || y < 0 || x >= screen_w_ || y >= screen_h_) return;

        uint32_t& dst = bits_[static_cast<size_t>(y) * static_cast<size_t>(screen_w_) + x];

        uint32_t sa = alpha_of(argb);
        if (sa == 0u)   return;                       // 完全透明：什么都不做
        if (sa == 255u) { dst = argb; return; }        // 完全不透明：直接覆盖

        // src 转预乘
        uint32_t sr = mul_div255(red_of(argb),   sa);
        uint32_t sg = mul_div255(green_of(argb), sa);
        uint32_t sb = mul_div255(blue_of(argb),  sa);

        // dst 已经是预乘值，只需乘以 (1 - srcA)
        uint32_t inv = 255u - sa;
        uint32_t dr = mul_div255(red_of(dst),   inv);
        uint32_t dg = mul_div255(green_of(dst), inv);
        uint32_t db = mul_div255(blue_of(dst),  inv);
        uint32_t da = mul_div255(alpha_of(dst), inv);

        uint32_t r = sr + dr, g = sg + dg, b = sb + db, a = sa + da;
        if (r > 255u) r = 255u;   // 两个预乘值相加可能溢出 8 位，钳位
        if (g > 255u) g = 255u;
        if (b > 255u) b = 255u;
        if (a > 255u) a = 255u;

        dst = (a << 24) | (r << 16) | (g << 8) | b;
    }

    // ------------------------------------------------------------------
    // band(): 闭区间矩形内的“覆盖写入”（put 而非 blend），供 draw_box 铺条带。
    // 内部仍走 put()，所以自动带上边界裁剪和预乘转换。
    // ------------------------------------------------------------------
    void band(int x1, int y1, int x2, int y2, uint32_t color) {
        for (int y = y1; y <= y2; ++y)
            for (int x = x1; x <= x2; ++x)
                put(x, y, color);
    }

    // ------------------------------------------------------------------
    // fixup_gdi_alpha(): 给刚被 GDI 写过文字的小矩形补上 alpha（详见 draw_text）。
    // 只处理 alpha==0 的像素：GDI 的输出必然是 alpha=0，而 buffer 里其他像素
    // 要么本来就透明（RGB 全 0，补完还是 0，无副作用），要么已被有效 alpha 覆盖。
    // ------------------------------------------------------------------
    void fixup_gdi_alpha(int x, int y, int w, int h) {
        if (!bits_ || w <= 0 || h <= 0) return;
        int x2 = x + w - 1, y2 = y + h - 1;
        if (x2 >= screen_w_) x2 = screen_w_ - 1;
        if (y2 >= screen_h_) y2 = screen_h_ - 1;
        for (int py = y; py <= y2; ++py) {
            if (py < 0 || py >= screen_h_) continue;
            // 行首指针：一次算好，行内用指针自增，比每像素重算索引便宜。
            uint32_t* row = bits_ + static_cast<size_t>(py) * static_cast<size_t>(screen_w_);
            for (int px = x; px <= x2; ++px) {
                if (px < 0 || px >= screen_w_) continue;
                uint32_t p = row[px];
                if (alpha_of(p) != 0u) continue;      // 非 GDI 笔触，跳过
                uint32_t r = red_of(p), g = green_of(p), b = blue_of(p);
                uint32_t coverage = r > g ? (r > b ? r : b) : (g > b ? g : b);
                if (coverage == 0u) continue;         // 黑色 = 背景，保持透明
                // RGB 已是预乘值，只需把 alpha 设为覆盖率
                row[px] = (coverage << 24) | (r << 16) | (g << 8) | b;
            }
        }
    }

    // GDI 对象的释放。顺序与创建相反，并且必须先把位图从 DC 上换下来
    // 才能 DeleteObject（被 SelectObject 选中的 GDI 对象删除会失败）。
    void cleanup_gdi() {
        if (mem_dc_ && hbitmap_) SelectObject(mem_dc_, old_bmp_);
        old_bmp_ = nullptr;

        if (hbitmap_) { DeleteObject(hbitmap_); hbitmap_ = nullptr; }
        if (mem_dc_)  { DeleteDC(mem_dc_);      mem_dc_ = nullptr; }
        // GetDC 拿到的 DC 必须用 ReleaseDC 还给系统，不能 DeleteDC。
        if (screen_dc_) { ReleaseDC(nullptr, screen_dc_); screen_dc_ = nullptr; }
        bits_ = nullptr;
    }

    // 窗口类名。进程内单例字符串，避免每次 show() 重新构造。
    static constexpr const wchar_t* kClassName = L"PyOverlayWnd";

    // ---- 所有 Win32 句柄：nullptr 一律表示“未持有”，是 hide() 幂等的基础 ----
    HWND      hwnd_        = nullptr;  // 分层窗口本体
    HDC       screen_dc_   = nullptr;  // 屏幕 DC（present 的 hdcDst）
    HDC       mem_dc_      = nullptr;  // 内存 DC（承载 hbitmap_，present 的 hdcSrc）
    HBITMAP   hbitmap_     = nullptr;  // 32bpp DIB Section，像素即 bits_
    HGDIOBJ   old_bmp_     = nullptr;  // mem_dc_ 上原先的位图（析构时要换回）
    uint32_t* bits_        = nullptr;  // 指向 DIB 像素的裸指针，勿手动 free
    int       screen_w_    = 0;        // 物理像素宽
    int       screen_h_    = 0;        // 物理像素高
    bool      shown_       = false;    // 是否已完成首帧显示
    bool      timer_started_ = false;  // timeBeginPeriod 是否已配对调用
    DWORD     owner_thread_ = 0;       // 创建 hwnd_ 的线程 ID
};

// ============================================================================
// pybind11 导出
// ============================================================================
PYBIND11_MODULE(overlay, m) {
    m.doc() = "极速透明叠加层(Win32 分层窗口 + 逐像素 alpha + 鼠标穿透 + 置顶)";

    py::class_<Overlay>(m, "Overlay")
        .def(py::init<>())
        .def("show",    &Overlay::show,
             "创建全屏分层窗口和像素缓冲；失败抛 RuntimeError")
        .def("hide",    &Overlay::hide,
             "销毁窗口并释放 GDI 资源，可重复调用")
        .def("clear",   &Overlay::clear,
             "整帧清空为全透明（每帧开头调用）")
        .def("draw_box", &Overlay::draw_box,
             py::arg("x1"), py::arg("y1"), py::arg("x2"), py::arg("y2"),
             py::arg("color") = 0xFFFF0000, py::arg("thickness") = 2,
             "空心矩形；坐标为闭区间（含右下端点），color=0xAARRGGBB")
        .def("fill_box", &Overlay::fill_box,
             py::arg("x1"), py::arg("y1"), py::arg("x2"), py::arg("y2"),
             py::arg("color") = 0x40FF0000,
             "实心矩形；color 的 alpha 会真正参与混合")
        .def("draw_line", &Overlay::draw_line,
             py::arg("x1"), py::arg("y1"), py::arg("x2"), py::arg("y2"),
             py::arg("color") = 0xFFFF0000, py::arg("thickness") = 1,
             "Bresenham 直线")
        .def("draw_text", &Overlay::draw_text,
             py::arg("x"), py::arg("y"), py::arg("text"),
             py::arg("color") = 0xFFFFFFFF, py::arg("size") = 16,
             "用系统字体光栅化文字（GDI 路径，每帧少量标签为宜）")
        .def("present", &Overlay::present,
             "把 buffer 提交给 DWM 合成，结束一帧")
        .def("width",    &Overlay::width)
        .def("height",   &Overlay::height);
}
