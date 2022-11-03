# Code structure

## Framework

The framework is a crude approximation of the [CMS](https://cms.cern/)
data processing software, [CMSSW](http://cms-sw.github.io/). CMSSW is
a rather generic framework to process independent chunks of data. In
CMS these chunks of data correspond to triggered proton-proton
collisions, and are called events. The events are processed by
modules, that in this project are all "producers" that can read
objects from the event, and insert new objects to the event (in CMSSW
there are, in addition, analyzers, that can only read objects from the
event, and filters, that can decide to stop processing of the event).

The modules form a DAG based on their data dependencies. The modules
are implemented in C++ (C++17 in general, CUDA code is in C++14). A
CMSSW job is configured in Python (this project does not provide
configuration mechanism).

The CMSSW framework is multi-threaded using [Threading Building Blocks
(TBB)](https://github.com/intel/tbb). An integral part of the
multi-threading is a concept of ["concurrent event
processor"](../src/cuda/bin/StreamSchedule.h) that we call "a stream"
(to disambiguate from CUDA streams, these streams are called "EDM
streams" from now on). An EDM stream processes one event at a time
("processing" meaning that each module in the DAG is run on the event
in some order respecting the data dependencies). A job may have
multiple EDM streams, in which case the EDM streams process their
events concurrently. Furthermore, modules processing the same event
that are independent in the DAG are also run concurrently. All this
potential concurrency is exposed as tasks to be run by the TBB task
scheduler. We do not make any assumptions on how TBB runs these tasks
in threads (e.g. number of EDM streams and number of threads may be
different). For more information on CMSSW framework see e.g.
- CMS TWiki pages
  - [SWGuideFrameWork](https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideFrameWork)
  - [MultithreadedFrameworkDesignDiscussions](https://twiki.cern.ch/twiki/bin/view/CMSPublic/MultithreadedFrameworkDesignDiscussions)
  - The CMS Offline WorkBook [WorkBook](https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBook)
  - The CMS Offline SW Guide [SWGuide](https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuide)
- Papers
  - [C D Jones et al 2014, J. Phys. Conf. Ser. 513 022034](https://iopscience.iop.org/article/10.1088/1742-6596/513/2/022034)
  - [C D Jones et al 2015, J. Phys. Conf. Ser. 664 072026](https://iopscience.iop.org/article/10.1088/1742-6596/664/7/072026)
  - [C D Jones et al 2017, J. Phys. Conf. Ser. 898 042008](https://iopscience.iop.org/article/10.1088/1742-6596/898/4/042008)

The processing time and memory requirements can vary a lot across
events. In addition, the filtering capability may affect which modules
in the DAG can be run.

The main approximations of the framework in this project with respect to CMSSW are
- producer modules only, and only stream and stream-ExternalWork producers
- modules and the event products have no labels, and therefore the event can hold only at most one product of each C++ type
- no run time configuration mechanism
- input data are fully read in memory at the beginning of the job
- EventSetup system has a single (implicit) record, one IOV, all products are read from binary dumps at the beginning of the job.

## Overall view on the use of CUDA

Our overall aims are to avoid blocking synchronization as much as
possible, and keep all processing units (CPU cores, GPUs) as busy as
we can doing useful work. It follows that we try to have all
operations (memory transfers, kernel calls) asynchronous with the use
of CUDA streams, and that we use callback functions
(`cudaStreamAddCallback()`) to notify the CMSSW framework when the
asynchronous work has finished.

We use a "caching allocator" (based on the one from
[CUB](https://nvlabs.github.io/cub/) library) for both device and
pinned host memory allocations. This approach allows us to amortize
the cost of the `cudaMalloc()`/`cudaFree()` etc, while being able to
easily re-use device/pinned host memory regions for temporary
workspaces, to avoid "conservative" overallocation of memory, and to
avoid constraining the scheduling of modules to multiple devices.

We use one CUDA stream for each EDM stream ("concurrent event") and
each linear chain of GPU modules that pass data from one to the other
in the device memory. In case of branches in the DAG of modules,
additional CUDA streams are used since there is sub-event concurrency
in the DAG that we want to expose to CUDA runtime.

For more information see [cms-sw/cmssw:`HeterogeneousCore/CUDACore/README.md`](https://github.com/cms-sw/cmssw/blob/CMSSW_11_1_0_pre4/HeterogeneousCore/CUDACore/README.md).
