# GitHub and Zenodo release guide

```bash
cd QT3-FIS-Quantum-Only-Public-Release
git init
git add .
git commit -m "Initial QT3-FIS quantum-only public release"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/QT3-FIS-Quantum-Only.git
git push -u origin main
```

Then create a GitHub release:

```text
v1.0.0
```

After that, connect the GitHub repository to Zenodo and archive the release to obtain a DOI.
