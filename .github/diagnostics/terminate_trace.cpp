#include <cstdlib>
#include <exception>
#include <execinfo.h>
#include <unistd.h>

static void trace_terminate() noexcept {
    constexpr char message[] = "METISCRAFT_NATIVE_TERMINATE_BACKTRACE\n";
    write(STDERR_FILENO, message, sizeof(message) - 1);
    void* frames[80];
    const int count = backtrace(frames, 80);
    backtrace_symbols_fd(frames, count, STDERR_FILENO);
    std::abort();
}

extern "C" void metiscraft_install_terminate_trace() {
    std::set_terminate(trace_terminate);
}
