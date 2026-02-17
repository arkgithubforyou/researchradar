output "instance_public_ip" {
  description = "Public IP address of the ResearchRadar VM"
  value       = oci_core_instance.app.public_ip
}

output "app_url" {
  description = "URL for the ResearchRadar web app"
  value       = var.domain_name != "" ? "https://${var.domain_name}" : "http://${oci_core_instance.app.public_ip}"
}

output "ssh_command" {
  description = "SSH command to connect to the VM"
  value       = "ssh opc@${oci_core_instance.app.public_ip}"
}

output "instance_ocid" {
  description = "OCID of the compute instance"
  value       = oci_core_instance.app.id
}
