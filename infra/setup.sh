#!/usr/bin/env bash
# One-time setup for the Four-Top Event Classifier's Azure deployment.
#
# Run this once, from your own machine, after `az login`. It:
#   1. Creates the resource group.
#   2. Deploys infra/main.bicep (Log Analytics, App Insights, ACR, Container
#      Apps environment + app with a placeholder image).
#   3. Registers a GitHub Actions app registration + federated credential,
#      so .github/workflows/deploy.yml can log in to Azure via OIDC --
#      no client secret is ever generated or stored.
#   4. Prints the `gh variable set` commands to wire the repo up. Run those
#      yourself (they're not run automatically, since this script doesn't
#      assume the GitHub CLI is installed or that you want it touching your
#      repo settings unattended).
#
# Safe to re-run: bicep deployments are idempotent, and the app registration
# step is skipped if AZURE_CLIENT_ID is already set below.
set -euo pipefail

# --- Fill these in ----------------------------------------------------------
LOCATION="${LOCATION:-eastus}"
RESOURCE_GROUP="${RESOURCE_GROUP:-fourtop-rg}"
NAME_PREFIX="${NAME_PREFIX:-fourtop}"
GITHUB_REPO="${GITHUB_REPO:-ryanwhitford/4top-event-classifier}"  # owner/repo
# -----------------------------------------------------------------------------

command -v az >/dev/null || { echo "Azure CLI not found. Install it: https://learn.microsoft.com/cli/azure/install-azure-cli" >&2; exit 1; }

echo "== Subscription =="
az account show --query "{name:name, id:id, tenantId:tenantId}" -o table
SUBSCRIPTION_ID=$(az account show --query id -o tsv)
TENANT_ID=$(az account show --query tenantId -o tsv)

echo "== Creating resource group: $RESOURCE_GROUP ($LOCATION) =="
az group create --name "$RESOURCE_GROUP" --location "$LOCATION" -o none

echo "== Deploying infra/main.bicep =="
DEPLOY_OUTPUT=$(az deployment group create \
  --resource-group "$RESOURCE_GROUP" \
  --template-file infra/main.bicep \
  --parameters namePrefix="$NAME_PREFIX" \
  -o json)

ACR_NAME=$(echo "$DEPLOY_OUTPUT" | python3 -c "import sys,json; print(json.load(sys.stdin)['properties']['outputs']['acrLoginServer']['value'].split('.')[0])")
CONTAINER_APP_NAME=$(echo "$DEPLOY_OUTPUT" | python3 -c "import sys,json; print(json.load(sys.stdin)['properties']['outputs']['containerAppName']['value'])")
CONTAINER_APP_FQDN=$(echo "$DEPLOY_OUTPUT" | python3 -c "import sys,json; print(json.load(sys.stdin)['properties']['outputs']['containerAppFqdn']['value'])")

echo "ACR:           $ACR_NAME"
echo "Container App: $CONTAINER_APP_NAME"
echo "URL (placeholder image for now): https://$CONTAINER_APP_FQDN"

echo "== Setting up GitHub Actions OIDC login (no client secret) =="
APP_NAME="github-oidc-$NAME_PREFIX"
EXISTING_APP_ID=$(az ad app list --display-name "$APP_NAME" --query "[0].appId" -o tsv 2>/dev/null || true)

if [ -n "$EXISTING_APP_ID" ]; then
  echo "App registration '$APP_NAME' already exists (appId=$EXISTING_APP_ID); reusing it."
  AZURE_CLIENT_ID="$EXISTING_APP_ID"
else
  AZURE_CLIENT_ID=$(az ad app create --display-name "$APP_NAME" --query appId -o tsv)
  az ad sp create --id "$AZURE_CLIENT_ID" -o none
  echo "Created app registration: $APP_NAME (appId=$AZURE_CLIENT_ID)"
fi

# Two role assignments, each scoped as tight as the workflow actually needs:
#   - Contributor on the resource group, for `az containerapp update` (Container
#     Apps doesn't have a narrower single built-in role for revision updates).
#   - AcrPush specifically on the registry, because ACR push/pull are *data
#     plane* operations -- Contributor alone does not include them, and
#     `docker push` would fail with an auth error without this.
ACR_ID=$(az acr show --name "$ACR_NAME" --resource-group "$RESOURCE_GROUP" --query id -o tsv)

# A freshly created service principal can take a few seconds to replicate
# through Azure AD; role assignments made immediately after can fail with a
# "principal not found" error even though everything is correct.
sleep 15

az role assignment create \
  --assignee "$AZURE_CLIENT_ID" \
  --role "Contributor" \
  --scope "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP" \
  -o none 2>/dev/null || echo "(Contributor role assignment already exists, skipping)"

az role assignment create \
  --assignee "$AZURE_CLIENT_ID" \
  --role "AcrPush" \
  --scope "$ACR_ID" \
  -o none 2>/dev/null || echo "(AcrPush role assignment already exists, skipping)"

# Federated credential: trusts GitHub Actions runs on the main branch of this
# repo specifically -- not arbitrary GitHub tokens.
az ad app federated-credential create \
  --id "$AZURE_CLIENT_ID" \
  --parameters "{
    \"name\": \"github-main-branch\",
    \"issuer\": \"https://token.actions.githubusercontent.com\",
    \"subject\": \"repo:${GITHUB_REPO}:ref:refs/heads/main\",
    \"audiences\": [\"api://AzureADTokenExchange\"]
  }" -o none 2>/dev/null || echo "(federated credential already exists, skipping)"

echo ""
echo "================================================================"
echo "Done. Set these as GitHub repo VARIABLES (not secrets -- none of"
echo "these values are sensitive on their own; OIDC needs no client secret):"
echo ""
echo "  gh variable set AZURE_CLIENT_ID       --body \"$AZURE_CLIENT_ID\""
echo "  gh variable set AZURE_TENANT_ID       --body \"$TENANT_ID\""
echo "  gh variable set AZURE_SUBSCRIPTION_ID --body \"$SUBSCRIPTION_ID\""
echo "  gh variable set AZURE_RESOURCE_GROUP  --body \"$RESOURCE_GROUP\""
echo "  gh variable set ACR_NAME              --body \"$ACR_NAME\""
echo "  gh variable set CONTAINER_APP_NAME    --body \"$CONTAINER_APP_NAME\""
echo ""
echo "Then push to main, or run the 'deploy' workflow manually from the"
echo "Actions tab, to replace the placeholder image with fourtop-serve."
echo "================================================================"
