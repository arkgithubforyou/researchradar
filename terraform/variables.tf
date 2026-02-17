variable "project" {
  description = "Project name used for resource naming"
  type        = string
  default     = "researchradar"
}

variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Deployment environment (dev, staging, prod)"
  type        = string
  default     = "prod"
}

# ── Networking ───────────────────────────────────────────────────────

variable "vpc_cidr" {
  description = "CIDR block for the VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "az_count" {
  description = "Number of availability zones to use"
  type        = number
  default     = 2
}

# ── ECS / Container ─────────────────────────────────────────────────

variable "container_port" {
  description = "Port the API container listens on"
  type        = number
  default     = 8000
}

variable "cpu" {
  description = "Fargate task CPU (256, 512, 1024, 2048, 4096)"
  type        = number
  default     = 1024
}

variable "memory" {
  description = "Fargate task memory in MiB"
  type        = number
  default     = 4096
}

variable "desired_count" {
  description = "Number of running ECS tasks"
  type        = number
  default     = 1
}

# ── Application ──────────────────────────────────────────────────────

variable "groq_api_key" {
  description = "Groq API key for cloud LLM"
  type        = string
  sensitive   = true
  default     = ""
}

variable "llm_backend" {
  description = "LLM backend to use (groq or ollama)"
  type        = string
  default     = "groq"
}

# ── Optional HTTPS ──────────────────────────────────────────────────

variable "domain_name" {
  description = "Custom domain name (e.g. researchradar.example.com). Leave empty to skip HTTPS."
  type        = string
  default     = ""
}

variable "certificate_arn" {
  description = "ARN of an ACM certificate for HTTPS. Leave empty to skip HTTPS."
  type        = string
  default     = ""
}
