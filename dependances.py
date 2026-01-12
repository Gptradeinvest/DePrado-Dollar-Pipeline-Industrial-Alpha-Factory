import subprocess
import sys

def install_dependencies():
    dependencies = [
        "pandas",
        "numpy",
        "matplotlib",
        "scipy",
        "scikit-learn",
        "statsmodels"
    ]
    
    print("Initialisation de l'Usine a Alpha : Installation des dependances...")
    
    for package in dependencies:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"SUCCES : {package} installe.")
        except Exception as e:
            print(f"ERREUR : Impossible d'installer {package} : {e}")

if __name__ == "__main__":
    install_dependencies()
    print("\nConfiguration terminee. Le pipeline est pret a l'emploi.")