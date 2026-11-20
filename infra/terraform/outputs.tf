# Outputs from the InvoiceOps Terraform deployment.

output "agent_url" {
  description = "Public URL of the InvoiceOps agent Cloud Run service"
  value       = google_cloud_run_v2_service.agent.uri
}

output "erp_url" {
  description = "Internal URL of the Mock ERP Cloud Run service"
  value       = google_cloud_run_v2_service.erp.uri
}

output "service_account_email" {
  description = "Email of the least-privilege service account"
  value       = google_service_account.invoiceops.email
}

output "bigquery_dataset" {
  description = "BigQuery dataset ID for policy clauses and eval runs"
  value       = google_bigquery_dataset.invoiceops.dataset_id
}
