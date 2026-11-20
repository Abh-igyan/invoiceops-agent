# Input variables for the InvoiceOps Terraform configuration.

variable "project_id" {
  description = "Google Cloud project ID"
  type        = string
}

variable "project_number" {
  description = "Google Cloud project number (numeric)"
  type        = string
}

variable "region" {
  description = "Default Google Cloud region for Cloud Run and Secret Manager"
  type        = string
  default     = "us-central1"
}

variable "bq_location" {
  description = "BigQuery dataset location"
  type        = string
  default     = "US"
}

variable "billing_account_id" {
  description = "Billing account ID for the budget alert"
  type        = string
}

variable "agent_image" {
  description = "Container image URI for the InvoiceOps agent Cloud Run service"
  type        = string
  default     = "us-central1-docker.pkg.dev/PROJECT_ID/invoiceops/agent:latest"
}

variable "erp_image" {
  description = "Container image URI for the Mock ERP Cloud Run service"
  type        = string
  default     = "us-central1-docker.pkg.dev/PROJECT_ID/invoiceops/erp:latest"
}

variable "monthly_budget_usd" {
  description = "Monthly spending budget in USD"
  type        = number
  default     = 10
}
