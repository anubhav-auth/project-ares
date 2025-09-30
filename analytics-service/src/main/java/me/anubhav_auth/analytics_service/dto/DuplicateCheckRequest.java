package me.anubhav_auth.analytics_service.dto;

import lombok.Data;

@Data
public class DuplicateCheckRequest {
    private String jobId;
    private String companyName;
    private String jobTitle;
    private String source;
}