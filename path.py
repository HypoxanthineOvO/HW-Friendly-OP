from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

# Modules
MODULES_ROOT = PROJECT_ROOT / "Modules"

# Testcases
TESTCASES_ROOT = PROJECT_ROOT / "Testcases"

if __name__ == "__main__":
    print("Project Root:", PROJECT_ROOT)
    print("Modules Root:", MODULES_ROOT)
    print("Testcases Root:", TESTCASES_ROOT)

    # Example usage
    if (MODULES_ROOT / "Attention").exists():
        print("Attention module exists.")
    else:
        print("Attention module does not exist.")
    
    if (TESTCASES_ROOT / "AttentionTest.py").exists():
        print("Attention test case exists.")
    else:
        print("Attention test case does not exist.")