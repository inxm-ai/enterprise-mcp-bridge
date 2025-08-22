#!/bin/bash
set -e

# Get the current version from pyproject.toml
current_version=$(grep -oP '(?<=version = ")[^"]*' ./app/pyproject.toml)
echo "Current version: $current_version"

# Ask the user for the type of release
read -p "What kind of version should be released? (patch, minor, major): " release_type

# Split the version into major, minor, and patch
IFS='.' read -r major minor patch <<< "$current_version"

# Increment the version based on the release type
case $release_type in
  patch)
    patch=$((patch + 1))
    ;;
  minor)
    minor=$((minor + 1))
    patch=0
    ;;
  major)
    major=$((major + 1))
    minor=0
    patch=0
    ;;
  *)
    echo "Invalid release type. Please choose patch, minor, or major."
    exit 1
    ;;
esac

# Construct the new version
new_version="$major.$minor.$patch"
echo "New version: $new_version"

# Update the version in pyproject.toml
sed -i "s/version = \"$current_version\"/version = \"$new_version\"/" ./app/pyproject.toml

# Commit the changes and create a tag
git add ./app/pyproject.toml
git commit -m "Releasing $new_version"
git tag "v$new_version"

# Push the changes and the tag to origin main
git push origin main
git push origin "v$new_version"

echo "Release $new_version has been pushed to origin main."
