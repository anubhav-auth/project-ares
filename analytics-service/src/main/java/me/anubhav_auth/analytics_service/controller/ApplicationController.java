package me.anubhav_auth.analytics_service.controller;

import me.anubhav_auth.analytics_service.entity.Application;
import me.anubhav_auth.analytics_service.repository.ApplicationRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController // Tells Spring this class defines REST API endpoints
@RequestMapping("/api/applications") // Base URL for all endpoints in this class
public class ApplicationController {

    @Autowired // Asks Spring to give us an instance of the repository
    private ApplicationRepository applicationRepository;

    /**
     * Endpoint to create a new application record. [cite: 111]
     */
    @PostMapping
    public Application createApplication(@RequestBody Application application) {
        return applicationRepository.save(application);
    }

    /**
     * Endpoint to get all stored application records. [cite: 114]
     */
    @GetMapping
    public List<Application> getAllApplications() {
        return applicationRepository.findAll();
    }

    /**
     * Endpoint to update the status of an existing application. [cite: 113]
     */
    @PutMapping("/{id}")
    public ResponseEntity<Application> updateApplicationStatus(
            @PathVariable Long id,
            @RequestBody Application applicationDetails) {
        return applicationRepository.findById(id)
                .map(application -> {
                    application.setStatus(applicationDetails.getStatus()); // Only update the status
                    Application updatedApplication = applicationRepository.save(application);
                    return ResponseEntity.ok(updatedApplication);
                })
                .orElse(ResponseEntity.notFound().build());
    }
}