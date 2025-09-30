package me.anubhav_auth.analytics_service.controller;

import lombok.RequiredArgsConstructor;
import me.anubhav_auth.analytics_service.dto.ApplicationStats;
import me.anubhav_auth.analytics_service.dto.BulkUpdateRequest;
import me.anubhav_auth.analytics_service.dto.DuplicateCheckRequest;
import me.anubhav_auth.analytics_service.entity.Application;
import me.anubhav_auth.analytics_service.entity.WorkflowError;
import me.anubhav_auth.analytics_service.repository.ApplicationRepository;
import me.anubhav_auth.analytics_service.repository.ErrorRepository;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.*;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api/applications")
@RequiredArgsConstructor
public class ApplicationController {

    private final ApplicationRepository applicationRepository;
    private final ErrorRepository errorRepository;

    @PostMapping
    public ResponseEntity<Application> createApplication(@RequestBody Application application) {
        // Check for duplicates BEFORE saving
        boolean isDuplicate = checkForDuplicate(
                application.getJobId(),
                application.getCompanyName(),
                application.getJobTitle()
        );

        if (isDuplicate) {
            // Return conflict status with existing application
            Optional<Application> existing = applicationRepository.findByJobId(application.getJobId());
            if (existing.isPresent()) {
                return ResponseEntity.status(HttpStatus.CONFLICT).body(existing.get());
            }
        }

        // Set default status if not provided
        if (application.getStatus() == null) {
            application.setStatus("NOTIFIED");
        }

        try {
            Application saved = applicationRepository.save(application);
            return ResponseEntity.status(HttpStatus.CREATED).body(saved);
        } catch (Exception e) {
            // Handle unique constraint violation
            if (e.getMessage().contains("duplicate") || e.getMessage().contains("unique")) {
                return ResponseEntity.status(HttpStatus.CONFLICT).build();
            }
            throw e;
        }
    }

    @PostMapping("/upsert")
    public ResponseEntity<Application> upsertApplication(@RequestBody Application application) {
        // Find existing by jobId or create new
        Optional<Application> existing = applicationRepository.findByJobId(application.getJobId());

        if (existing.isPresent()) {
            Application existingApp = existing.get();
            // Update only if status is still NOTIFIED (hasn't been processed)
            if ("NOTIFIED".equals(existingApp.getStatus())) {
                existingApp.setJobTitle(application.getJobTitle());
                existingApp.setCompanyName(application.getCompanyName());
                existingApp.setDossier(application.getDossier());
                existingApp.setSource(application.getSource());
                return ResponseEntity.ok(applicationRepository.save(existingApp));
            } else {
                // Already processed, don't update
                return ResponseEntity.status(HttpStatus.NOT_MODIFIED).body(existingApp);
            }
        } else {
            // New application
            if (application.getStatus() == null) {
                application.setStatus("NOTIFIED");
            }
            return ResponseEntity.status(HttpStatus.CREATED).body(applicationRepository.save(application));
        }
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
    public ResponseEntity<Map<String, Object>> checkDuplicate(
            @RequestParam(required = false) String jobId,
            @RequestParam(required = false) String companyName,
            @RequestParam(required = false) String jobTitle,
            @RequestParam(defaultValue = "24") int hoursBack) {

        Map<String, Object> response = new HashMap<>();

        // Priority 1: Check by unique jobId
        if (jobId != null && !jobId.isEmpty()) {
            boolean exists = applicationRepository.existsByJobId(jobId);
            response.put("exists", exists);
            response.put("checkMethod", "jobId");

            if (exists) {
                Optional<Application> existing = applicationRepository.findByJobId(jobId);
                existing.ifPresent(app -> {
                    response.put("existingId", app.getId());
                    response.put("existingStatus", app.getStatus());
                });
            }
            return ResponseEntity.ok(response);
        }

        // Priority 2: Check by company + job title within time window
        if (companyName != null && jobTitle != null) {
            Instant cutoffTime = Instant.now().minus(hoursBack, ChronoUnit.HOURS);

            boolean exists = applicationRepository.existsByCompanyNameAndJobTitleAndCreatedAtAfter(
                    companyName, jobTitle, cutoffTime
            );

            response.put("exists", exists);
            response.put("checkMethod", "companyAndTitle");
            response.put("hoursBack", hoursBack);

            if (exists) {
                List<Application> existingApps = applicationRepository
                        .findByCompanyNameAndJobTitleAndCreatedAtAfter(companyName, jobTitle, cutoffTime);
                if (!existingApps.isEmpty()) {
                    Application mostRecent = existingApps.get(0);
                    response.put("existingId", mostRecent.getId());
                    response.put("existingStatus", mostRecent.getStatus());
                    response.put("duplicateCount", existingApps.size());
                }
            }
            return ResponseEntity.ok(response);
        }

        // No valid parameters provided
        response.put("error", "Please provide either jobId or both companyName and jobTitle");
        return ResponseEntity.badRequest().body(response);
    }

    @PostMapping("/check-duplicates-bulk")
    public ResponseEntity<Map<String, Object>> checkDuplicatesBulk(
            @RequestBody List<Map<String, String>> jobs) {

        Map<String, Object> response = new HashMap<>();
        List<Map<String, Object>> results = new ArrayList<>();

        Instant cutoffTime = Instant.now().minus(48, ChronoUnit.HOURS);

        for (Map<String, String> job : jobs) {
            String jobId = job.get("jobId");
            String companyName = job.get("companyName");
            String jobTitle = job.get("jobTitle");

            Map<String, Object> checkResult = new HashMap<>();
            checkResult.put("jobId", jobId);

            // Check by jobId first (most reliable)
            if (jobId != null && applicationRepository.existsByJobId(jobId)) {
                checkResult.put("isDuplicate", true);
                checkResult.put("reason", "jobId");
            }
            // Then check by company + title
            else if (companyName != null && jobTitle != null) {
                boolean exists = applicationRepository
                        .existsByCompanyNameAndJobTitleAndCreatedAtAfter(
                                companyName, jobTitle, cutoffTime
                        );
                checkResult.put("isDuplicate", exists);
                checkResult.put("reason", exists ? "companyAndTitle" : "unique");
            } else {
                checkResult.put("isDuplicate", false);
                checkResult.put("reason", "insufficient_data");
            }

            results.add(checkResult);
        }

        long duplicateCount = results.stream()
                .filter(r -> (Boolean) r.get("isDuplicate"))
                .count();

        response.put("results", results);
        response.put("totalChecked", jobs.size());
        response.put("duplicateCount", duplicateCount);
        response.put("uniqueCount", jobs.size() - duplicateCount);

        return ResponseEntity.ok(response);
    }

    @PostMapping("/standardize-and-save")
    public ResponseEntity<Application> standardizeAndSave(@RequestBody Application application) {
        // Standardize company name
        if (application.getCompanyName() != null) {
            application.setCompanyName(standardizeCompanyName(application.getCompanyName()));
        }

        // Check for duplicates after standardization
        boolean isDuplicate = checkForDuplicate(
                application.getJobId(),
                application.getCompanyName(),
                application.getJobTitle()
        );

        if (isDuplicate) {
            Optional<Application> existing = applicationRepository.findByJobId(application.getJobId());
            if (existing.isPresent()) {
                return ResponseEntity.status(HttpStatus.CONFLICT).body(existing.get());
            }
        }

        // Set default status
        if (application.getStatus() == null) {
            application.setStatus("NOTIFIED");
        }

        return ResponseEntity.status(HttpStatus.CREATED).body(applicationRepository.save(application));
    }

    private String standardizeCompanyName(String name) {
        // Remove common suffixes
        name = name.replaceAll("(?i)\\s+(inc|llc|ltd|limited|corp|corporation|company|co)\\.*$", "");
        // Normalize spacing
        name = name.replaceAll("\\s+", " ").trim();
        // Title case
        return Arrays.stream(name.split(" "))
                .map(word -> word.substring(0, 1).toUpperCase() + word.substring(1).toLowerCase())
                .collect(Collectors.joining(" "));
    }

    @PostMapping("/check-duplicates-batch")
    public ResponseEntity<Map<String, Object>> checkDuplicatesBatch(
            @RequestBody List<DuplicateCheckRequest> requests) {

        Map<String, Object> results = new HashMap<>();
        List<String> duplicates = new ArrayList<>();
        List<String> unique = new ArrayList<>();

        for (DuplicateCheckRequest request : requests) {
            boolean isDuplicate = checkForDuplicate(
                    request.getJobId(),
                    request.getCompanyName(),
                    request.getJobTitle()
            );

            if (isDuplicate) {
                duplicates.add(request.getJobId());
            } else {
                unique.add(request.getJobId());
            }
        }

        results.put("totalChecked", requests.size());
        results.put("duplicates", duplicates);
        results.put("unique", unique);
        results.put("duplicateCount", duplicates.size());
        results.put("uniqueCount", unique.size());

        return ResponseEntity.ok(results);
    }

    private boolean checkForDuplicate(String jobId, String companyName, String jobTitle) {
        // Check by jobId first
        if (jobId != null && applicationRepository.existsByJobId(jobId)) {
            return true;
        }

        // Check by company and title in last 48 hours
        if (companyName != null && jobTitle != null) {
            Instant cutoff = Instant.now().minus(48, ChronoUnit.HOURS);
            return applicationRepository.existsByCompanyNameAndJobTitleAndCreatedAtAfter(
                    companyName, jobTitle, cutoff
            );
        }

        return false;
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