# ── Oracle Cloud Infrastructure (OCI) — Always Free Tier ────────────
#
# Deploys ResearchRadar on an A1.Flex ARM VM (free forever).
# Requires: OCI account with Always Free resources available.

variable "tenancy_ocid" {
  description = "OCI tenancy OCID (from OCI console → Profile → Tenancy)"
  type        = string
}

variable "user_ocid" {
  description = "OCI user OCID (from OCI console → Profile → My Profile)"
  type        = string
}

variable "fingerprint" {
  description = "API key fingerprint (from OCI console → Profile → API Keys)"
  type        = string
}

variable "private_key_path" {
  description = "Path to the OCI API private key PEM file"
  type        = string
}

variable "region" {
  description = "OCI region (e.g., us-ashburn-1, us-phoenix-1, eu-frankfurt-1)"
  type        = string
  default     = "us-ashburn-1"
}

variable "compartment_ocid" {
  description = "OCI compartment OCID (use tenancy_ocid for root compartment)"
  type        = string
}

# ── VM sizing (Always Free limits: 4 OCPU, 24 GB total) ────────────

variable "instance_ocpus" {
  description = "Number of OCPUs for the A1 instance (max 4 for free tier)"
  type        = number
  default     = 2
}

variable "instance_memory_gb" {
  description = "Memory in GB for the A1 instance (max 24 for free tier)"
  type        = number
  default     = 12
}

variable "boot_volume_gb" {
  description = "Boot volume size in GB (up to 200 GB free)"
  type        = number
  default     = 50
}

# ── Application ─────────────────────────────────────────────────────

variable "groq_api_key" {
  description = "Groq API key for cloud LLM"
  type        = string
  sensitive   = true
  default     = ""
}

variable "ssh_public_key" {
  description = "SSH public key for VM access (paste the full key string)"
  type        = string
}

variable "domain_name" {
  description = "Optional domain name pointing to the VM's public IP. Leave empty to use IP directly."
  type        = string
  default     = ""
}
