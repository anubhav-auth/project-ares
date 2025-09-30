package me.anubhav_auth.analytics_service.controller;

import lombok.RequiredArgsConstructor;
import me.anubhav_auth.analytics_service.dto.ApplicationStats;
import me.anubhav_auth.analytics_service.dto.BulkUpdateRequest;
import me.anubhav_auth.analytics_service.entity.Application;
import me.anubhav_auth.analytics_service.entity.WorkflowError;
import me.anubhav_auth.analytics_service.repository.ApplicationRepository;
import me.anubhav_auth.analytics_service.repository.ErrorRepository;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/applications")
@RequiredArgsConstructor
public class ApplicationController {

    private final ApplicationRepository applicationRepository;
    private final ErrorRepository errorRepository;

    @PostMapping
    public Application createApplication(@RequestBody Application application) {
        // Set default status if not provided
        if (application.getStatus() == null) {
            application.setStatus("NOTIFIED");
        }
        return applicationRepository.save(application);
    }

    @GetMapping
    public List<Application> getAllApplications() {
        return applicationRepository.findAll();
    }

    @GetMapping("/{id}")
    public ResponseEntity<Application> getApplicationById(@PathVariable Long id) {
        return applicationRepository.findById(id)
                .map(ResponseEntity::ok)
                .orElse(ResponseEntity.notFound().build());
    }

    @PutMapping("/{id}")
    public ResponseEntity<Application> updateApplicationStatus(
            @PathVariable Long id,
            @RequestBody Application applicationDetails) {
        return applicationRepository.findById(id)
                .map(application -> {
                    // Update status
                    if (applicationDetails.getStatus() != null) {
                        application.setStatus(applicationDetails.getStatus());
                    }
                    // Update action type if provided
                    if (applicationDetails.getActionType() != null) {
                        application.setActionType(applicationDetails.getActionType());
                    }
                    // Update cover letter if provided
                    if (applicationDetails.getGeneratedCoverLetter() != null) {
                        application.setGeneratedCoverLetter(applicationDetails.getGeneratedCoverLetter());
                    }
                    Application updatedApplication = applicationRepository.save(application);
                    return ResponseEntity.ok(updatedApplication);
                })
                .orElse(ResponseEntity.notFound().build());
    }

    @GetMapping("/check-duplicate")
    public ResponseEntity<Map<String, Boolean>> checkDuplicate(
            @RequestParam String jobKey) {

        // Check if similar job was applied to recently
        Instant oneDayAgo = Instant.now().minus(1, ChronoUnit.DAYS);

        // Create a composite key from company and job title
        String[] parts = jobKey.split("_");
        String companyName = parts.length > 0 ? parts[0] : "";
        String jobTitle = parts.length > 1 ? parts[1] : "";

        boolean exists = applicationRepository.existsByCompanyNameAndJobTitleAndCreatedAtAfter(
                companyName, jobTitle, oneDayAgo
        );

        Map<String, Boolean> response = new HashMap<>();
        response.put("exists", exists);

        return ResponseEntity.ok(response);
    }

    @GetMapping("/stats")
    public ResponseEntity<ApplicationStats> getStats() {
        ApplicationStats stats = new ApplicationStats();

        stats.setTotalApplications(applicationRepository.count());
        stats.setAppliedCount(applicationRepository.countByStatus("APPLIED"));
        stats.setPendingCount(applicationRepository.countByStatus("WAITING_APPROVAL"));
        stats.setFailedCount(applicationRepository.countByStatus("FAILED"));
        stats.setNotifiedCount(applicationRepository.countByStatus("NOTIFIED"));

        // Success rate
        if (stats.getTotalApplications() > 0) {
            stats.setSuccessRate(
                    (double) stats.getAppliedCount() / stats.getTotalApplications() * 100
            );
        }

        // Get recent applications
        Instant sevenDaysAgo = Instant.now().minus(7, ChronoUnit.DAYS);
        stats.setRecentApplicationsCount(
                applicationRepository.countByCreatedAtAfter(sevenDaysAgo)
        );

        return ResponseEntity.ok(stats);
    }

    @PostMapping("/bulk-update")
    public ResponseEntity<List<Application>> bulkUpdateStatus(
            @RequestBody BulkUpdateRequest request) {

        List<Application> updated = new ArrayList<>();

        for (Long id : request.getIds()) {
            applicationRepository.findById(id).ifPresent(app -> {
                app.setStatus(request.getStatus());
                if (request.getActionType() != null) {
                    app.setActionType(request.getActionType());
                }
                updated.add(applicationRepository.save(app));
            });
        }

        return ResponseEntity.ok(updated);
    }

    @DeleteMapping("/{id}")
    public ResponseEntity<Void> deleteApplication(@PathVariable Long id) {
        if (applicationRepository.existsById(id)) {
            applicationRepository.deleteById(id);
            return ResponseEntity.noContent().build();
        }
        return ResponseEntity.notFound().build();
    }

    @GetMapping("/by-status/{status}")
    public ResponseEntity<List<Application>> getApplicationsByStatus(@PathVariable String status) {
        return ResponseEntity.ok(applicationRepository.findByStatus(status));
    }

    @GetMapping("/by-company/{companyName}")
    public ResponseEntity<List<Application>> getApplicationsByCompany(@PathVariable String companyName) {
        return ResponseEntity.ok(applicationRepository.findByCompanyNameContainingIgnoreCase(companyName));
    }
}