Owner: Hari Krishna Kancharla.

primary.zip preserves the exact pinned vLLM files named in SOURCE.json. The files
are primary-source test data under their original Apache-2.0 license. The checkout
already retains that license at licenses/vLLM-Apache-2.0.txt. The source files keep
their original license headers. No engine or device code is imported by tests.

Tests verify each complete file's SHA-256 and Git blob identity, then compile only
selected Python method fragments with ordinary host dependencies. These checks
exercise the observer and its exact call shapes. They establish no native engine,
GPU, live condition, physical memory read or deployment acceptance.
