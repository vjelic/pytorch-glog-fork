#ifndef AMD_FRAMEWORKS_INTERNAL_TENSORDUMP_H
#define AMD_FRAMEWORKS_INTERNAL_TENSORDUMP_H

#include <string>
#include <fstream>

namespace {

const char* cast_dtype(caffe2::TypeMeta t_dtype)
{
#define CAST_TYPE(aname, dtname) if (t_dtype == at::aname) return #dtname;
  CAST_TYPE(kByte, uint8);
  CAST_TYPE(kUInt16, uint16);
  CAST_TYPE(kUInt32, uint32);
  CAST_TYPE(kUInt64, uint64);
  CAST_TYPE(kChar, int8);
  CAST_TYPE(kShort, int16);
  CAST_TYPE(kInt, int32);
  CAST_TYPE(kLong, int64);
  CAST_TYPE(kHalf, float16);
  CAST_TYPE(kFloat, float32);
  CAST_TYPE(kBFloat16, bfloat16);
  return "unknown";
#undef CAST_TYPE
}

template<typename Tensor>
void metadump(const Tensor& t, std::fstream& fout)
{
  fout << R"zzz({ "sizes":[)zzz";
  const char* spacer = "";
  for (int s : t.sizes()) {
    fout << spacer << s;
    spacer = ", ";
  }
  fout << R"zzz(], "strides": [)zzz";
  spacer = "";
  for (int s : t.strides()) {
    fout << spacer << s;
    spacer = ", ";
  }
  fout << R"zzz(], "dtype": ")zzz";
  fout << cast_dtype(t.dtype());
  fout << R"zzz(", "offset": )zzz";
  fout << reinterpret_cast<const char*>(t.data_ptr()) - reinterpret_cast<const char*>(t.storage().data_ptr().get());
  fout << R"zzz(, "nbytes": )zzz";
  fout << t.nbytes();
  fout << R"zzz(})zzz";
}

template<typename Tensor>
void datadump(const Tensor& t, std::fstream& fout)
{
  fout.write(reinterpret_cast<char*>(t.storage().data_ptr().get()),
             t.storage().nbytes());
}

template<typename Tensor>
void tensordump(const Tensor& t, const std::string& basename)
{
  {
    std::fstream fout(basename + ".json", std::ios::out | std::ios::trunc);
    metadump(t, fout);
  }
  {
    std::fstream fout(basename + ".tdata", std::ios::out | std::ios::binary | std::ios::trunc);
    datadump(t, fout);
  }
}
}

#endif
