// test_driver.cpp
#include "Source Code/Tests/tests.h"
#include <iostream>

int main() {
    if (TAPA::Tests::run_all_tests()) {
        std::cout << "All tests passed!" << std::endl;
        return 0;
    } else {
        std::cout << "Some tests failed." << std::endl;
        return 1;
    }
}