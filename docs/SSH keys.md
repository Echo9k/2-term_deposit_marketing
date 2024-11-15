Steps to add the SSH keys
You might need this if you're getting the `Permission denied (publickey)` error.

### 1. Verify if You Have an SSH Key
Check if you have an SSH key by listing the contents of your SSH directory:

```bash
ls -al ~/.ssh
```

You should see a pair of files like `id_rsa` and `id_rsa.pub` (or other names ending in `.pub`). The `.pub` file is your public key, which you’ll upload to GitHub.

### 2. Generate an SSH Key (if you don’t have one)
If you don’t see an SSH key file, you’ll need to generate a new one. Use this command:

```bash
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
```

Replace `"your_email@example.com"` with the email associated with your GitHub account. When prompted, you can save the key in the default location (press Enter) and create a passphrase if desired.

### 3. Start the SSH Agent and Add Your Key
To use your SSH key, you’ll need to start the SSH agent and add your private key:

```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_rsa
```

Replace `id_rsa` with your private key file name if it’s different.

### 4. Add Your SSH Key to GitHub
You’ll now need to add the SSH key to your GitHub account:

1. Copy the SSH key to your clipboard:
   ```bash
   cat ~/.ssh/id_rsa.pub
   ```

2. Go to GitHub, navigate to **Settings** > **SSH and GPG keys** > **New SSH key**.
3. Paste your SSH key and give it a descriptive title.

### 5. Test the SSH Connection to GitHub
To confirm your SSH connection to GitHub is working:

```bash
ssh -T git@github.com
```

If successful, you’ll see a message like:

```
Hi username! You've successfully authenticated, but GitHub does not provide shell access.
```

### 6. Try `git pull` Again
Now, go back to your project directory and try running:

```bash
git pull
```

This should successfully connect to GitHub and pull the latest changes if your SSH key is configured correctly.