import subprocess

def initialize_dvc():
    # Use the full path to the DVC executable
    dvc_executable = r'C:\Users\username\AppData\Roaming\Python\Python311\Scripts\dvc.exe'
    
    # Run the DVC init command
    process = subprocess.run(
        [dvc_executable, 'init'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Print standard output and error
    print("stdout:", process.stdout)
    print("stderr:", process.stderr)

def main():
    print("Starting the model version control script...")
    print("Initializing DVC...")
    initialize_dvc()

if __name__ == "__main__":
    main()
