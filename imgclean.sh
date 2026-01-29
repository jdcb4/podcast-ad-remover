#!/bin/bash

# Configuration
OWNER="akeslo"
PACKAGE="podcast-ad-remover"

echo "🧹 Fetching untagged versions for $OWNER/$PACKAGE..."

# 1. Fetch versions
# 2. Filter for untagged versions ONLY
# 3. Sort by date (newest first)
# 4. Skip the first 2 versions (Keeps the very latest internal manifests)
# 5. Loop through the rest and delete
gh api "/users/$OWNER/packages/container/$PACKAGE/versions" --paginate -q '.' \
  | jq -r 'map(select(.metadata.container.tags | length == 0)) | sort_by(.created_at) | reverse | .[2:] | .[].id' \
  | while read -r version_id; do
      echo "Deleting version ID: $version_id..."
      
      # The '> /dev/null' silences the output so you don't have to hit 'q'
      gh api --method DELETE \
        "/users/$OWNER/packages/container/$PACKAGE/versions/$version_id" > /dev/null
  done

echo "Cleanup complete!"
