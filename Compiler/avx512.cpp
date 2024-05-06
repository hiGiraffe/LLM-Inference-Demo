#include <iostream>

int main() {
#ifdef __AVX512BF16__
    std::cout << "AVX512 BF16 is supported!" << std::endl;
#else
    std::cout << "AVX512 BF16 is not supported." << std::endl;
#endif

    return 0;
}