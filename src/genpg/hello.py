import torch as tch


def main():
    dev = tch.device("cuda")
    x = tch.randn((4, 3, 2), device=dev)
    x2 = x**2
    print(f"Hello from gen-playground:\n{x2}")


if __name__ == "__main__":
    main()
