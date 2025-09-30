package me.anubhav_auth.analytics_service.dto;

import lombok.Data;
import java.util.List;

@Data
public class BulkUpdateRequest {
    private List<Long> ids;
    private String status;
    private String actionType;
}