// best-rk3588.rknn model bilgisi cikarma araci.
//
// Bu araci CIHAZDA (RK3588, gercek NPU driver + librknnrt.so ile) derleyip
// calistir. Cikti, vision-app inference kodu icin gereken TUM bilgileri
// verir: girdi/cikti tensor sayisi, isim, sekil, format (NHWC/NCHW), veri
// tipi, kuantizasyon tipi (affine asymmetric), zero_point, scale, stride'lar
// ve SDK/driver versiyonu.
//
// Derleme (cihazda, aarch64):
//   g++ -std=c++17 -O2 model_info_tool.cpp \
//       -I ../third_party/librknn_api/include \
//       -L ../third_party/librknn_api/aarch64 -lrknnrt \
//       -Wl,-rpath,../third_party/librknn_api/aarch64 \
//       -o model_info_tool
//
// Calistirma:
//   ./model_info_tool ../models/best-rk3588.rknn

#include "rknn_api.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

namespace {

std::vector<uint8_t> readFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    std::streamsize size = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<uint8_t> buf(static_cast<size_t>(size));
    if (!f.read(reinterpret_cast<char*>(buf.data()), size)) return {};
    return buf;
}

void printAttr(const rknn_tensor_attr& a) {
    printf("  [%u] name=%-24s n_dims=%u dims=(", a.index, a.name, a.n_dims);
    for (uint32_t i = 0; i < a.n_dims; ++i) {
        printf("%u%s", a.dims[i], (i + 1 < a.n_dims) ? "," : "");
    }
    printf(")\n");
    printf("       fmt=%s type=%s qnt_type=%s zp=%d scale=%.8f\n",
           get_format_string(a.fmt), get_type_string(a.type),
           get_qnt_type_string(a.qnt_type), a.zp, a.scale);
    printf("       n_elems=%u size=%u w_stride=%u h_stride=%u size_with_stride=%u "
           "pass_through=%u\n",
           a.n_elems, a.size, a.w_stride, a.h_stride, a.size_with_stride,
           a.pass_through);
}

// index gecerli degilse RKNN_SUCC disi doner; bu durumda query'i sessizce atla.
void printQueryVariant(rknn_context ctx, int query_id, const char* label, uint32_t n,
                        bool is_input) {
    printf("-- %s --\n", label);
    for (uint32_t i = 0; i < n; ++i) {
        rknn_tensor_attr attr{};
        attr.index = i;
        int ret = rknn_query(ctx, static_cast<rknn_query_cmd>(query_id), &attr,
                              sizeof(attr));
        if (ret != RKNN_SUCC) {
            printf("  [%u] sorgu desteklenmiyor / basarisiz (ret=%d)\n", i, ret);
            continue;
        }
        printAttr(attr);
    }
    (void)is_input;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "kullanim: %s <model.rknn>\n", argv[0]);
        return 1;
    }

    auto model_data = readFile(argv[1]);
    if (model_data.empty()) {
        fprintf(stderr, "model dosyasi okunamadi: %s\n", argv[1]);
        return 1;
    }

    rknn_context ctx = 0;
    int ret = rknn_init(&ctx, model_data.data(),
                         static_cast<uint32_t>(model_data.size()), 0, nullptr);
    if (ret != RKNN_SUCC) {
        fprintf(stderr, "rknn_init basarisiz: ret=%d\n", ret);
        return 1;
    }

    rknn_sdk_version sdk_ver{};
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &sdk_ver, sizeof(sdk_ver));
    if (ret == RKNN_SUCC) {
        printf("== SDK / Driver Versiyonu ==\n");
        printf("  api_version=%s driver_version=%s\n", sdk_ver.api_version,
               sdk_ver.drv_version);
    }

    rknn_input_output_num io_num{};
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) {
        fprintf(stderr, "RKNN_QUERY_IN_OUT_NUM basarisiz: ret=%d\n", ret);
        rknn_destroy(ctx);
        return 1;
    }
    printf("\n== Girdi/Cikti Sayisi ==\n  n_input=%u n_output=%u\n\n", io_num.n_input,
           io_num.n_output);

    // Runtime'a rknn_inputs_set / rknn_outputs_get ile veri gonderirken/alirken
    // gorecegin (donusum sonrasi) mantiksal format.
    printQueryVariant(ctx, RKNN_QUERY_INPUT_ATTR, "GIRDI ATTR (mantiksal)",
                       io_num.n_input, true);
    printQueryVariant(ctx, RKNN_QUERY_OUTPUT_ATTR, "CIKTI ATTR (mantiksal)",
                       io_num.n_output, false);

    // NPU'nun ic donanim/native formati (zero-copy / pass_through=1 icin gerekli).
    printQueryVariant(ctx, RKNN_QUERY_NATIVE_INPUT_ATTR, "GIRDI ATTR (native/NPU)",
                       io_num.n_input, true);
    printQueryVariant(ctx, RKNN_QUERY_NATIVE_OUTPUT_ATTR, "CIKTI ATTR (native/NPU)",
                       io_num.n_output, false);

    rknn_mem_size mem_size{};
    ret = rknn_query(ctx, RKNN_QUERY_MEM_SIZE, &mem_size, sizeof(mem_size));
    if (ret == RKNN_SUCC) {
        printf("\n== Bellek Kullanimi ==\n");
        printf("  total_weight_size=%u total_internal_size=%u total_dma_allocated_size=%llu\n",
               mem_size.total_weight_size, mem_size.total_internal_size,
               static_cast<unsigned long long>(mem_size.total_dma_allocated_size));
    }

    rknn_destroy(ctx);
    return 0;
}
