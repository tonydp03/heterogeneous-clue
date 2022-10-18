#ifndef SYCLDataFormats_Plugin_Wrapper_h
#define SYCLDataFormats_Plugin_Wrapper_h

namespace cms {
  namespace sycltools {
    template <typename T, typename P>
    class PluginWrapper {
    public:
      template <typename... Args>
      explicit PluginWrapper(Args&&... args) : obj_{std::forward<Args>(args)...} {}
      T const& get() const { return obj_; }

    private:
      T obj_;
    };

  }  // namespace sycltools
}  // namespace cms

#endif