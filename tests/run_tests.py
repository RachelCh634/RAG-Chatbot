import subprocess

if __name__ == "__main__":
    print("Running all tests...")
    result = subprocess.run(["python", "-m", "pytest", "-v"], capture_output=True, text=True)

    print("\n🧪 Pytest Output:")
    print(result.stdout)

    if result.stderr:
        print("\n⚠ Errors:")
        print(result.stderr)