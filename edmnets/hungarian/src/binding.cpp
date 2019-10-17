#include "hungarian.h"

PYBIND11_MODULE(bindings, m) {
    m.def("hungarian_reorder", [](np_array<float> cost, np_array<float> edms,
                                  const np_array<std::int32_t> &types, const np_array<std::int32_t> &typesRec) {
        if(cost.shape(1) != cost.shape(2)) throw std::invalid_argument("requires square matrix for costs");
        auto b = cost.shape(0);
        auto n = cost.shape(1);

        //if(edms.shape(0) != b) throw std::invalid_argument(fmt::format("batchsize mismatch "
        //                                                               "cost {} and edms {}", b, edms.shape(0)));
        //if(edms.shape(1) != n) throw std::invalid_argument(fmt::format("edms n = {} != {} = cost n", edms.shape(1), n));
        //if(edms.shape(1) != edms.shape(2)) throw std::invalid_argument("edms not square matrices (B, N, N, ...)");

        auto costPtr = cost.mutable_data(0);

        np_array<std::int32_t> typesCopy ({b, n});
        {
            std::copy(types.data(0), types.data(0) + b * n, typesCopy.mutable_data());
        }
        {
            // reconstructed types pointer for setting up cost matrix
            auto tptr = types.data(0);
            auto tptrRec = typesRec.data(0);

            #pragma omp parallel for default(none) firstprivate(b, n, costPtr, tptr, tptrRec)
            for(decltype(b) i = 0; i < b; ++i) {
                auto btptr = tptr + i * n;
                auto btptrRec = tptrRec + i * n;
                for(decltype(n) j = 0; j < n; ++j) {
                    auto typeJ = *(btptr + j);
                    for(decltype(n) k = 0; k < n; ++k) {
                        auto typeK = *(btptrRec + k);
                        if(typeJ != typeK) {
                            *(costPtr + i*n*n + j*n + k) = 10000.f + edmnets::random::normal(0.f, 1.f);
                        }
                    }
                }
            }
        }

        auto edmPtr = edms.mutable_data(0);
        auto typesCopyPtr = typesCopy.mutable_data(0);
        std::unique_ptr<float[]> edmCopy = std::unique_ptr<float[]>(new float[b * n * n]);
        std::copy(edmPtr, edmPtr + b * n * n, edmCopy.get());
        auto edmCopyPtr = edmCopy.get();
        auto tptr = types.data(0);
        #pragma omp parallel default(none) firstprivate(b, n, costPtr, edmPtr, edmCopyPtr, typesCopyPtr, tptr)
        {
            std::unique_ptr<long[]> rowOut = std::unique_ptr<long[]>(new long[n]);
            std::unique_ptr<long[]> colOut = std::unique_ptr<long[]>(new long[n]);
            #pragma omp for
            for (decltype(b) i = 0; i < b; ++i) {
                auto costo = costPtr + i * n * n;
                edmnets::hungarian::asp::asp(n, edmnets::hungarian::Array2DAccess<float>(costo, n),
                                             colOut.get(), rowOut.get());

                auto edm = edmPtr + i * n * n;
                auto copy = edmCopyPtr + i * n * n;
                auto t = tptr + i * n;
                auto tc = typesCopyPtr + i*n;

                for(decltype(n) j = 0; j < n; ++j) {
                    *(tc + j) = *(t + *(rowOut.get() + j));
                    for(decltype(n) k = 0; k < n; ++k) {
                        *(edm + k + j * n) = *(copy + *(rowOut.get() + k) + *(rowOut.get() + j) * n);
                    }
                }
            }
        }
        return std::make_tuple(edms, typesCopy);
    }, py::arg("cost"), py::arg("edms"), py::arg("types"), py::arg("types_rec"));
}
