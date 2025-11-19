#!/usr/bin/env bash
set -e

GREEN="\033[1;32m"
YELLOW="\033[1;33m"
NC="\033[0m"

echo -e "${GREEN}=== Building minigrad ===${NC}"

rm -rf build
mkdir -p build
cd build

echo -e "${YELLOW}Configuring...${NC}"
cmake -DCMAKE_BUILD_TYPE=Release ..

echo -e "${YELLOW}Building...${NC}"
cmake --build . -- -j$(nproc)

echo -e "${GREEN}=== Build complete ===${NC}"
