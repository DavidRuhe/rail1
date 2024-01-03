import subprocess


def git_detached():
    # Get the output of 'git status'
    git_status_output = subprocess.getoutput("git status")
    return "HEAD detached" in git_status_output


def check_git_detached():
    if git_detached():
        raise RuntimeError(f"git is a detached HEAD. Please checkout a branch.")

