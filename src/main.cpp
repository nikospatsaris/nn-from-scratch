// Trains the network. Uses MNIST if the data is present, spirals otherwise.

#include <chrono>
#include <cstdio>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

#include "data.hpp"
#include "network.hpp"

namespace {

struct Config {
    std::size_t epochs = 10;
    std::size_t batchSize = 64;
    double learningRate = 0.1;
    std::size_t hidden = 128;
    std::size_t trainLimit = 0;  // 0 = all
    bool forceSpirals = false;
    std::string exportPath;  // empty = do not export
};

void printUsage() {
    std::printf(
        "usage: train [options]\n"
        "  --epochs N        training passes over the data (default 10)\n"
        "  --batch N         mini-batch size (default 64)\n"
        "  --lr F            learning rate (default 0.1)\n"
        "  --hidden N        hidden layer width (default 128)\n"
        "  --limit N         use only the first N training samples\n"
        "  --spirals         use the generated spiral set, ignoring MNIST\n"
        "  --export FILE     write the trained weights to FILE as JSON\n");
}

bool parseArgs(int argc, char** argv, Config& cfg) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&](std::size_t& out) {
            if (i + 1 >= argc) return false;
            out = static_cast<std::size_t>(std::stoul(argv[++i]));
            return true;
        };
        if (arg == "--epochs" && next(cfg.epochs)) continue;
        if (arg == "--batch" && next(cfg.batchSize)) continue;
        if (arg == "--hidden" && next(cfg.hidden)) continue;
        if (arg == "--limit" && next(cfg.trainLimit)) continue;
        if (arg == "--lr" && i + 1 < argc) { cfg.learningRate = std::stod(argv[++i]); continue; }
        if (arg == "--spirals") { cfg.forceSpirals = true; continue; }
        if (arg == "--export" && i + 1 < argc) { cfg.exportPath = argv[++i]; continue; }
        if (arg == "--help" || arg == "-h") { printUsage(); return false; }
        std::printf("unknown option: %s\n\n", arg.c_str());
        printUsage();
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    Config cfg;
    if (!parseArgs(argc, argv, cfg)) return 1;

    const std::string trainImages = "data/train-images-idx3-ubyte";
    const std::string trainLabels = "data/train-labels-idx1-ubyte";
    const std::string testImages = "data/t10k-images-idx3-ubyte";
    const std::string testLabels = "data/t10k-labels-idx1-ubyte";

    nn::Dataset train, test;
    std::string datasetName;

    const bool haveMnist = nn::fileExists(trainImages) && nn::fileExists(trainLabels) &&
                           nn::fileExists(testImages) && nn::fileExists(testLabels);

    try {
        if (haveMnist && !cfg.forceSpirals) {
            datasetName = "MNIST handwritten digits";
            train = nn::loadMnist(trainImages, trainLabels, cfg.trainLimit);
            test = nn::loadMnist(testImages, testLabels, 0);
        } else {
            datasetName = "generated spirals (3 classes)";
            if (!cfg.forceSpirals) {
                std::printf("MNIST not found in data/ — run scripts/get_mnist.sh to use it.\n");
                std::printf("Falling back to the generated spiral dataset.\n\n");
            }
            train = nn::makeSpirals(400, 3, 7);
            test = nn::makeSpirals(150, 3, 99);
        }
    } catch (const std::exception& e) {
        std::printf("failed to load data: %s\n", e.what());
        return 1;
    }

    const std::size_t inputSize = train.X.cols();
    const std::size_t outputSize = train.numClasses;

    nn::Network net({inputSize, cfg.hidden, outputSize});

    std::printf("Dataset      %s\n", datasetName.c_str());
    std::printf("Train/test   %zu / %zu samples\n", train.y.size(), test.y.size());
    std::printf("Network      %zu -> %zu -> %zu\n", inputSize, cfg.hidden, outputSize);
    std::printf("Batch / LR   %zu / %g\n\n", cfg.batchSize, cfg.learningRate);

    std::printf("%6s %12s %12s %12s %10s\n",
                "epoch", "train loss", "train acc", "test acc", "seconds");

    const auto wallStart = std::chrono::steady_clock::now();

    for (std::size_t epoch = 1; epoch <= cfg.epochs; ++epoch) {
        const auto t0 = std::chrono::steady_clock::now();
        const double loss = net.trainEpoch(train.X, train.y, cfg.batchSize, cfg.learningRate);
        const auto t1 = std::chrono::steady_clock::now();

        const double trainAcc = net.accuracy(train.X, train.y);
        const double testAcc = net.accuracy(test.X, test.y);
        const double secs = std::chrono::duration<double>(t1 - t0).count();

        std::printf("%6zu %12.4f %11.2f%% %11.2f%% %10.2f\n",
                    epoch, loss, trainAcc * 100.0, testAcc * 100.0, secs);
        std::fflush(stdout);
    }

    const double total = std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - wallStart).count();
    std::printf("\nFinal test accuracy: %.2f%%   (total %.1fs)\n",
                net.accuracy(test.X, test.y) * 100.0, total);

    if (!cfg.exportPath.empty()) {
        try {
            net.save(cfg.exportPath);
            std::printf("Weights written to %s\n", cfg.exportPath.c_str());
        } catch (const std::exception& e) {
            std::printf("export failed: %s\n", e.what());
            return 1;
        }
    }
    return 0;
}
