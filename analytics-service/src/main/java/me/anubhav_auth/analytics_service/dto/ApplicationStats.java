package me.anubhav_auth.analytics_service.dto;

import lombok.Data;

@Data
public class ApplicationStats {
    private long totalApplications;
    private long appliedCount;
    private long pendingCount;
    private long failedCount;
    private long notifiedCount;
    private double successRate;
    private long recentApplicationsCount;

    // Additional useful stats
    private long todayCount;
    private long weekCount;
    private long monthCount;
}