import subprocess


def run_bash_command(command: str) -> str:
    try:
        # Execute the command and capture the output
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Return the standard output
        return result.stdout
    except subprocess.CalledProcessError as e:
        # Return the standard error if the command fails
        return e.stderr
