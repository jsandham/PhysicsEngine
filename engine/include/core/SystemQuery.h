#ifndef SYSTEM_QUERY_H__
#define SYSTEM_QUERY_H__

#include <assert.h>

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) \
    || defined(__WIN64) && !defined(__CYGWIN__)
#include <windows.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace PhysicsEngine
{
    enum class ProcessorArch
    {
        AMD64,
        ARM,
        ARM64,
        IntelItanium,
        Intel,
        Unknown
    };

    constexpr auto ProcessorArchToString(ProcessorArch arch)
    {
        switch (arch)
        {
        case ProcessorArch::AMD64:
            return "AMD64"; 
        case ProcessorArch::ARM:
            return "ARM";
        case ProcessorArch::ARM64:
            return "AMR64";
        case ProcessorArch::IntelItanium:
            return "Intel Itanium";
        case ProcessorArch::Intel:
            return "Intel";
        default:
            return "Unknown";
        }
    }

    struct CPUInfo
    {
        ProcessorArch arch;
        int numCpuCores;
        int pageSize;
        bool openmpEnabled;
        int openmp_max_threads;
    };

    void queryCpuInfo(CPUInfo* info)
    {
        assert(info != nullptr);

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) \
    || defined(__WIN64) && !defined(__CYGWIN__)
        SYSTEM_INFO sysinfo;

        GetSystemInfo(&sysinfo);

        switch (sysinfo.wProcessorArchitecture)
        {
        case PROCESSOR_ARCHITECTURE_AMD64:
        {
            info->arch = ProcessorArch::AMD64;
            break;
        }
        case PROCESSOR_ARCHITECTURE_ARM:
        {
            info->arch = ProcessorArch::ARM;
            break;
        }
        case PROCESSOR_ARCHITECTURE_ARM64:
        {
            info->arch = ProcessorArch::ARM64;
            break;
        }
        case PROCESSOR_ARCHITECTURE_IA64:
        {
            info->arch = ProcessorArch::IntelItanium;
            break;
        }
        case PROCESSOR_ARCHITECTURE_INTEL:
        {
            info->arch = ProcessorArch::Intel;
            break;
        }
        default:
            info->arch = ProcessorArch::Unknown;
        }

        info->numCpuCores = static_cast<int>(sysinfo.dwNumberOfProcessors);
        info->pageSize = static_cast<int>(sysinfo.dwPageSize);
#endif // windows

        info->openmpEnabled = false;
        info->openmp_max_threads = 0;
#ifdef _OPENMP
        info->openmpEnabled = true;
        info->openmp_max_threads = omp_get_max_threads();
#endif
    }
} // namespace rocalution

#endif