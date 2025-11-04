#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <iomanip>

// --- Naiwna jednoprzebiegowa metoda ---
std::pair<double, double> variance_naive(const std::vector<double>& x) {
    int n = x.size();
    double sum = 0.0, sum_sq = 0.0;
    for (double v : x) {
        sum += v;
        sum_sq += v * v;
    }
    double mean = sum / n;
    double var = (sum_sq / n) - (mean * mean);
    return {mean, var};
}

// --- Dwuprzebiegowa metoda ---
std::pair<double, double> variance_two_pass(const std::vector<double>& x) {
    int n = x.size();
    double mean = 0.0;
    for (double v : x) mean += v;
    mean /= n;

    double var = 0.0;
    for (double v : x) {
        double diff = v - mean;
        var += diff * diff;
    }
    var /= n;
    return {mean, var};
}

// --- Jednoprzebiegowa metoda Welforda ---
std::pair<double, double> variance_welford(const std::vector<double>& x) {
    int n = 0;
    double mean = 0.0, M2 = 0.0;

    for (double v : x) {
        n++;
        double delta = v - mean;
        mean += delta / n;
        M2 += delta * (v - mean);
    }

    if (n == 0) return {mean, 0.0};

    double var = M2 / n;
    return {mean, var};
}

// --- Funkcja pomocnicza do wypisywania wyników ---
void test(const std::vector<double>& x, const std::string& label) {
    std::cout << std::fixed << std::setprecision(5);
    std::cout << "\nTest dla: " << label << std::endl;
    auto [m1, v1] = variance_naive(x);
    auto [m2, v2] = variance_two_pass(x);
    auto [m3, v3] = variance_welford(x);

    std::cout << "Naiwna jednoprzebieg.: mean = " << m1 << ", var = " << v1 << std::endl;
    std::cout << "Dwuprzebiegowa:        mean = " << m2 << ", var = " << v2 << std::endl;
    std::cout << "Welford:               mean = " << m3 << ", var = " << v3 << std::endl;
}

int main() {
    std::vector<double> base = {4, 7, 13, 16};
    std::vector<double> data1 = base;
    std::vector<double> data2, data3;

    for (double v : base) {
        data2.push_back(1e8 + v);
        data3.push_back(1e9 + v);
    }

    test(data1, "(4, 7, 13, 16)");
    test(data2, "1e8 + (4, 7, 13, 16)");
    test(data3, "1e9 + (4, 7, 13, 16)");

    return 0;
}
