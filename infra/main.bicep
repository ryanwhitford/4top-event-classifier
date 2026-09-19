// Infrastructure for the Four-Top Event Classifier's serving API.
//
// Log Analytics + Application Insights (monitoring) -> Container Registry
// (image storage) -> Container Apps Environment + Container App (compute),
// pulling images with a user-assigned managed identity rather than the
// registry's admin credentials.
//
// This deploys the *scaffolding* with a placeholder image; .github/workflows
// deploy.yml points the Container App at the real fourtop-serve image on
// every push to main. Run this once via infra/setup.sh, then let CI take
// over deployments.
//
// Deliberately NOT provisioned here: a managed online endpoint (Azure ML) --
// that bills by the hour whether or not it's serving traffic. Container Apps
// on the Consumption plan scales to zero and only costs money while a
// request is actually being handled, which matters on a $100 student credit.

@description('Azure region for all resources.')
param location string = resourceGroup().location

@description('Short, unique-ish prefix used to build resource names (lowercase alphanumeric).')
param namePrefix string = 'fourtop'

@description('Container image to deploy initially. CI overwrites this on every push to main via `az containerapp update --image`.')
param initialImage string = 'mcr.microsoft.com/azuredocs/containerapps-helloworld:latest'

@description('Container listens on this port inside the container.')
param containerPort int = 8000

@description('Max replicas. Min is fixed at 0 so the app scales to zero (and costs nothing) when idle.')
@minValue(1)
@maxValue(5)
param maxReplicas int = 2

var uniqueSuffix = uniqueString(resourceGroup().id)
var acrName = toLower('${namePrefix}acr${uniqueSuffix}')
var logAnalyticsName = '${namePrefix}-logs'
var appInsightsName = '${namePrefix}-insights'
var containerAppEnvName = '${namePrefix}-env'
var containerAppName = '${namePrefix}-serve'
var identityName = '${namePrefix}-serve-identity'

resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: logAnalyticsName
  location: location
  properties: {
    sku: { name: 'PerGB2018' }
    retentionInDays: 30 // keep this modest -- ingestion/retention both cost money past the free tier
  }
}

resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: appInsightsName
  location: location
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: logAnalytics.id
    IngestionMode: 'LogAnalytics'
  }
}

resource acr 'Microsoft.ContainerRegistry/registries@2023-11-01-preview' = {
  name: acrName
  location: location
  sku: { name: 'Basic' }
  properties: {
    adminUserEnabled: false // pulls happen via the managed identity below, not an admin password
  }
}

resource identity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: identityName
  location: location
}

// AcrPull, scoped to just this registry -- the minimum the Container App needs.
var acrPullRoleId = '7f951dda-4ed3-4680-a7ca-43fe172d538d'
resource acrPullAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(acr.id, identity.id, acrPullRoleId)
  scope: acr
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', acrPullRoleId)
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

resource containerAppEnv 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: containerAppEnvName
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalytics.properties.customerId
        sharedKey: logAnalytics.listKeys().primarySharedKey
      }
    }
  }
}

resource containerApp 'Microsoft.App/containerApps@2024-03-01' = {
  name: containerAppName
  location: location
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${identity.id}': {}
    }
  }
  properties: {
    managedEnvironmentId: containerAppEnv.id
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: true
        targetPort: containerPort
        transport: 'auto'
      }
      registries: [
        {
          server: acr.properties.loginServer
          identity: identity.id
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'fourtop-serve'
          image: initialImage
          resources: {
            cpu: json('0.5')
            memory: '1Gi'
          }
          env: [
            {
              name: 'APPLICATIONINSIGHTS_CONNECTION_STRING'
              value: appInsights.properties.ConnectionString
            }
          ]
        }
      ]
      scale: {
        minReplicas: 0 // scale to zero between requests -- this is the cost control that matters most
        maxReplicas: maxReplicas
      }
    }
  }
  dependsOn: [
    acrPullAssignment
  ]
}

output acrLoginServer string = acr.properties.loginServer
output containerAppName string = containerApp.name
output containerAppFqdn string = containerApp.properties.configuration.ingress.fqdn
output containerAppIdentityClientId string = identity.properties.clientId
