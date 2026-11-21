# InvoiceOps Agent — Terraform Infrastructure (Section 8)
# Two-service Cloud Run topology, least-privilege IAM, Secret Manager, BigQuery, budget alert.

terraform {
  required_version = ">= 1.5.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# ---------------------------------------------------------------------------
# Service Account — least-privilege (ADR-006)
# ---------------------------------------------------------------------------
resource "google_service_account" "invoiceops" {
  account_id   = "invoiceops-agent"
  display_name = "InvoiceOps Agent Service Account"
  description  = "Least-privilege SA for Cloud Run agent and ERP services"
}

locals {
  sa_roles = [
    "roles/bigquery.dataEditor",
    "roles/bigquery.jobUser",
    "roles/secretmanager.secretAccessor",
    "roles/cloudtrace.agent",
    "roles/logging.logWriter",
  ]
}

resource "google_project_iam_member" "invoiceops_roles" {
  for_each = toset(local.sa_roles)
  project  = var.project_id
  role     = each.value
  member   = "serviceAccount:${google_service_account.invoiceops.email}"
}

# ---------------------------------------------------------------------------
# Secret Manager — GEMINI_API_KEY
# ---------------------------------------------------------------------------
resource "google_secret_manager_secret" "gemini_api_key" {
  secret_id = "gemini-api-key"
  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_iam_member" "invoiceops_access" {
  secret_id = google_secret_manager_secret.gemini_api_key.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.invoiceops.email}"
}

# ---------------------------------------------------------------------------
# BigQuery — dataset + policy_clauses + eval_runs tables
# ---------------------------------------------------------------------------
resource "google_bigquery_dataset" "invoiceops" {
  dataset_id    = "invoiceops"
  friendly_name = "InvoiceOps Agent Data"
  description   = "Policy clauses (RAG) and evaluation run metrics"
  location      = var.bq_location
}

resource "google_bigquery_table" "policy_clauses" {
  dataset_id          = google_bigquery_dataset.invoiceops.dataset_id
  table_id            = "policy_clauses"
  deletion_protection = false
  schema              = <<-EOF
  [
    {"name": "clause_id",      "type": "STRING",  "mode": "REQUIRED"},
    {"name": "doc_id",         "type": "STRING",  "mode": "REQUIRED"},
    {"name": "doc_title",      "type": "STRING",  "mode": "NULLABLE"},
    {"name": "version",        "type": "STRING",  "mode": "REQUIRED"},
    {"name": "effective_from", "type": "DATE",    "mode": "NULLABLE"},
    {"name": "status",         "type": "STRING",  "mode": "REQUIRED"},
    {"name": "topic",          "type": "STRING",  "mode": "NULLABLE"},
    {"name": "text",           "type": "STRING",  "mode": "REQUIRED"},
    {"name": "embedding",      "type": "FLOAT64", "mode": "REPEATED"}
  ]
  EOF
}

resource "google_bigquery_table" "eval_runs" {
  dataset_id          = google_bigquery_dataset.invoiceops.dataset_id
  table_id            = "eval_runs"
  deletion_protection = false
  schema              = <<-EOF
  [
    {"name": "run_id",                "type": "STRING",    "mode": "REQUIRED"},
    {"name": "experiment_version",    "type": "STRING",    "mode": "REQUIRED"},
    {"name": "config_version",        "type": "STRING",    "mode": "REQUIRED"},
    {"name": "git_sha",              "type": "STRING",    "mode": "REQUIRED"},
    {"name": "split_name",           "type": "STRING",    "mode": "REQUIRED"},
    {"name": "rep_index",            "type": "INT64",     "mode": "REQUIRED"},
    {"name": "case_id",              "type": "STRING",    "mode": "REQUIRED"},
    {"name": "failure_mode",         "type": "STRING",    "mode": "REQUIRED"},
    {"name": "is_prompt_injection",  "type": "BOOL",      "mode": "REQUIRED"},
    {"name": "ground_truth_decision","type": "STRING",    "mode": "REQUIRED"},
    {"name": "predicted_decision",   "type": "STRING",    "mode": "REQUIRED"},
    {"name": "is_correct",           "type": "BOOL",      "mode": "REQUIRED"},
    {"name": "weighted_error_cost",  "type": "FLOAT64",   "mode": "REQUIRED"},
    {"name": "citation_valid",       "type": "BOOL",      "mode": "REQUIRED"},
    {"name": "retrieval_recall_at_k","type": "FLOAT64",   "mode": "REQUIRED"},
    {"name": "retrieval_mrr",        "type": "FLOAT64",   "mode": "REQUIRED"},
    {"name": "injection_resisted",   "type": "BOOL",      "mode": "NULLABLE"},
    {"name": "failure_category",     "type": "STRING",    "mode": "REQUIRED"},
    {"name": "latency_ms",           "type": "FLOAT64",   "mode": "REQUIRED"},
    {"name": "input_tokens",         "type": "INT64",     "mode": "REQUIRED"},
    {"name": "output_tokens",        "type": "INT64",     "mode": "REQUIRED"},
    {"name": "estimated_cost_usd",   "type": "FLOAT64",   "mode": "REQUIRED"},
    {"name": "llm_calls",            "type": "INT64",     "mode": "REQUIRED"},
    {"name": "tool_calls",           "type": "INT64",     "mode": "REQUIRED"},
    {"name": "erp_retries",          "type": "INT64",     "mode": "REQUIRED"},
    {"name": "resolved_model",       "type": "STRING",    "mode": "REQUIRED"},
    {"name": "explanation",          "type": "STRING",    "mode": "NULLABLE"},
    {"name": "timestamp_utc",        "type": "TIMESTAMP", "mode": "REQUIRED"}
  ]
  EOF
}

# ---------------------------------------------------------------------------
# Cloud Run v2 — Agent Service (public)
# ---------------------------------------------------------------------------
resource "google_cloud_run_v2_service" "agent" {
  name     = "invoiceops-agent"
  location = var.region

  template {
    service_account = google_service_account.invoiceops.email

    containers {
      image = var.agent_image

      ports {
        container_port = 8080
      }

      env {
        name = "GEMINI_API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.gemini_api_key.secret_id
            version = "latest"
          }
        }
      }

      env {
        name  = "INVOICEOPS_TIMEOUT_SECONDS"
        value = "120"
      }

      env {
        name  = "MOCK_ERP_BASE_URL"
        value = "${google_cloud_run_v2_service.erp.uri}/mcp/"
      }

      resources {
        limits = {
          cpu    = "1"
          memory = "1Gi"
        }
      }

      startup_probe {
        http_get {
          path = "/health"
        }
        initial_delay_seconds = 5
        period_seconds        = 5
        failure_threshold     = 3
      }

      liveness_probe {
        http_get {
          path = "/health"
        }
        period_seconds = 30
      }
    }

    scaling {
      min_instance_count = 0
      max_instance_count = 2
    }

    timeout = "300s"
  }
}

# Allow unauthenticated access to the agent service (demo)
resource "google_cloud_run_v2_service_iam_member" "agent_public" {
  name     = google_cloud_run_v2_service.agent.name
  location = var.region
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# ---------------------------------------------------------------------------
# Cloud Run v2 — Mock ERP Service (internal only)
# ---------------------------------------------------------------------------
resource "google_cloud_run_v2_service" "erp" {
  name     = "invoiceops-erp"
  location = var.region

  template {
    service_account = google_service_account.invoiceops.email

    containers {
      image = var.erp_image

      ports {
        container_port = 8001
      }

      resources {
        limits = {
          cpu    = "0.5"
          memory = "512Mi"
        }
      }

      startup_probe {
        http_get {
          path = "/health"
        }
        initial_delay_seconds = 3
        period_seconds        = 5
        failure_threshold     = 3
      }
    }

    scaling {
      min_instance_count = 0
      max_instance_count = 1
    }

    timeout = "60s"
  }
}

# ERP service accessible only by the agent service account (least-privilege)
resource "google_cloud_run_v2_service_iam_member" "erp_agent_access" {
  name     = google_cloud_run_v2_service.erp.name
  location = var.region
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.invoiceops.email}"
}

# ---------------------------------------------------------------------------
# Billing Budget Alert ($10/month at 50/80/100%)
# ---------------------------------------------------------------------------
resource "google_billing_budget" "invoiceops" {
  billing_account = var.billing_account_id
  display_name    = "InvoiceOps Monthly Budget"

  budget_filter {
    projects = ["projects/${var.project_number}"]
  }

  amount {
    specified_amount {
      currency_code = "USD"
      units         = var.monthly_budget_usd
    }
  }

  dynamic "threshold_rules" {
    for_each = [0.5, 0.8, 1.0]
    content {
      threshold_percent = threshold_rules.value
      spend_basis       = "CURRENT_SPEND"
    }
  }
}
