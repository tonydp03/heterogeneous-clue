#ifndef HeterogeneousCore_SYCLUtilities_interface_device_shared_ptr_h
#define HeterogeneousCore_SYCLUtilities_interface_device_shared_ptr_h

#include <functional>
#include <memory>
#include <optional>

#include <CL/sycl.hpp>

namespace cms {
  namespace sycltools {
    namespace device {
      namespace impl {
        class SharedDeviceDeleter {
        public:
          SharedDeviceDeleter() = default;  // for edm::Wrapper
          SharedDeviceDeleter(sycl::queue stream) : stream_{stream} {}

          void operator()(void *ptr) {
            if (stream_) {
              sycl::free(ptr, *stream_);
            }
          }

        private:
          std::optional<sycl::queue> stream_;
        };
      }  // namespace impl

      template <typename T>
      using shared_ptr = std::shared_ptr<T>;

      namespace impl {
        template <typename T>
        struct make_device_shared_selector {
          using non_array = cms::sycltools::device::shared_ptr<T>;
        };
        template <typename T>
        struct make_device_shared_selector<T[]> {
          using unbounded_array = cms::sycltools::device::shared_ptr<T[]>;
        };
        template <typename T, size_t N>
        struct make_device_shared_selector<T[N]> {
          struct bounded_array {};
        };
      }  // namespace impl
    }    // namespace device

    template <typename T>
    typename device::impl::make_device_shared_selector<T>::non_array make_device_shared(sycl::queue stream) {
      static_assert(std::is_trivially_constructible<T>::value,
                    "Allocating with non-trivial constructor on the device memory is not supported");
      void *mem = sycl::malloc_device(sizeof(T), stream);
      return typename device::impl::make_device_shared_selector<T>::non_array{
          reinterpret_cast<T *>(mem), device::impl::SharedDeviceDeleter{stream}};
    }

    template <typename T>
    typename device::impl::make_device_shared_selector<T>::unbounded_array make_device_shared(size_t n,
                                                                                              sycl::queue stream) {
      using element_type = typename std::remove_extent<T>::type;
      static_assert(std::is_trivially_constructible<element_type>::value,
                    "Allocating with non-trivial constructor on the device memory is not supported");
      void *mem = sycl::malloc_device(n * sizeof(element_type), stream);
      return typename device::impl::make_device_shared_selector<T>::unbounded_array{
          reinterpret_cast<element_type *>(mem), device::impl::SharedDeviceDeleter{stream}};
    }

    template <typename T, typename... Args>
    typename device::impl::make_device_shared_selector<T>::bounded_array make_device_shared(Args &&...) = delete;

    // No check for the trivial constructor, make it clear in the interface
    template <typename T>
    typename device::impl::make_device_shared_selector<T>::non_array make_device_shared_uninitialized(
        sycl::queue stream) {
      void *mem = sycl::malloc_device(sizeof(T), stream);
      return typename device::impl::make_device_shared_selector<T>::non_array{
          reinterpret_cast<T *>(mem), device::impl::SharedDeviceDeleter{stream}};
    }

    template <typename T>
    typename device::impl::make_device_shared_selector<T>::unbounded_array make_device_shared_uninitialized(
        size_t n, sycl::queue stream) {
      using element_type = typename std::remove_extent<T>::type;
      void *mem = sycl::malloc_device(n * sizeof(element_type), stream);
      return typename device::impl::make_device_shared_selector<T>::unbounded_array{
          reinterpret_cast<element_type *>(mem), device::impl::SharedDeviceDeleter{stream}};
    }

    template <typename T, typename... Args>
    typename device::impl::make_device_shared_selector<T>::bounded_array make_device_shared_uninitialized(Args &&...) =
        delete;
  }  // namespace sycltools
}  // namespace cms

#endif
