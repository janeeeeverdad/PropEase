// Global analysis functions
window.analyzePDFContent = async function() {
    console.log("⭐⭐⭐ GLOBAL analyzePDFContent FUNCTION CALLED ⭐⭐⭐");

    try {
        // Get the file ID from the URL
        const pathParts = window.location.pathname.split('/');
        const fileId = pathParts[pathParts.length - 1];
        console.log(`File ID: ${fileId}`);

       // Show loading indicator
        const loadingDiv = document.createElement('div');
        loadingDiv.id = 'global-loading-indicator';
        loadingDiv.style.position = 'fixed';
        loadingDiv.style.top = '50%';
        loadingDiv.style.left = '50%';
        loadingDiv.style.transform = 'translate(-50%, -50%)';
        loadingDiv.style.background = 'linear-gradient(135deg, #a18cd1 0%, #fbc2eb 100%)';
        loadingDiv.style.color = '#4a0072'; 
        loadingDiv.style.padding = '25px 30px';
        loadingDiv.style.borderRadius = '20px';
        loadingDiv.style.boxShadow = '0 8px 30px rgba(128, 90, 213, 0.3)';
        loadingDiv.style.zIndex = '9999';
        loadingDiv.style.fontFamily = '"Comfortaa", sans-serif';
        loadingDiv.innerHTML = `
            <div style="text-align: center;">
                <div style="margin-bottom: 12px;">
                    <i class="fas fa-spinner fa-spin" style="font-size: 26px; color: #6a1b9a;"></i>
                </div>
                <div>
                    <strong style="font-size: 16px;">Analyzing your document...</strong><br>
                    <small style="color: #4a0072;">Please wait while we process your content.</small>
                </div>
            </div>
        `;
        document.body.appendChild(loadingDiv);

        // Get the extracted text
        const extractedText = document.querySelector('#extracted-text')?.innerText || "";
        console.log(`Extracted text length: ${extractedText.length}`);

        // Get the speech text
        const speechText = window.transcriptText || "";
        console.log(`Speech text length: ${speechText.length}`);

        // Make the API call
        console.log(`Making API call to /analyze_content/${fileId}`);
        const response = await fetch(`/analyze_content/${fileId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                extracted_text: extractedText,
                speech_text: speechText
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        // Parse the response
        const results = await response.json();
        console.log("Analysis results:", results);

        // Update the display
        updateAnalysisDisplay(results);

        // Show the popup
        const popup = document.getElementById('popup');
        if (popup) {
            popup.style.display = 'flex';
            console.log("Showing popup");
        }
        // Generate titles
        if (typeof window.generateTitles === 'function') {
            window.generateTitles();
        }
        return results;
    } catch (error) {
        console.error("Error in analyzePDFContent:", error);
        alert("An error occurred during analysis. Please try again.");
    } finally {
        // Remove loading indicator
        const loadingDiv = document.getElementById('global-loading-indicator');
        if (loadingDiv) {
            loadingDiv.remove();
        }
    }
};

// Global function to update the analysis display
window.updateAnalysisDisplay = function(results) {
    console.log("⭐⭐⭐ GLOBAL updateAnalysisDisplay FUNCTION CALLED ⭐⭐⭐");
    console.log("Results:", results);

    if (!results) {
        console.error("No results to display");
        return;
    }

    // Update speech similarity
    const speechProgressBar = document.getElementById('progressSpeech');
    const speechLabel = document.getElementById('speechSimilarityLabel');
    if (speechProgressBar && speechLabel && results.speech_similarity !== undefined) {
        const similarityValue = Math.round(results.speech_similarity);
        speechProgressBar.style.width = `${similarityValue}%`;
        speechProgressBar.style.display = 'block';
        speechProgressBar.style.backgroundColor = '#d8b4f8';
        speechLabel.textContent = `${similarityValue}%`;
        console.log(`Updated speech similarity to ${similarityValue}%`);
    }

    // Update thesis similarity
    const thesisProgressBar = document.getElementById('progressThesis');
    const thesisLabel = document.getElementById('thesisSimilarityLabel');
    if (thesisProgressBar && thesisLabel && results.thesis_similarity !== undefined) {
        const thesisSimilarityValue = Math.round(results.thesis_similarity);
        thesisProgressBar.style.width = `${thesisSimilarityValue}%`;
        thesisProgressBar.style.display = 'block';
        thesisProgressBar.style.backgroundColor = '#d8b4f8';
        thesisLabel.textContent = `${thesisSimilarityValue}%`;
        console.log(`Updated thesis similarity to ${thesisSimilarityValue}%`);
    }

    // Update discrepancies
    const discrepancyList = document.getElementById('discrepancyList');
    if (discrepancyList && results.missed_keypoints && results.added_keypoints) {
        discrepancyList.innerHTML = '';

        // Add missed keypoints
        results.missed_keypoints.forEach(point => {
            const li = document.createElement('li');
            li.innerHTML = `🔻 <strong>Missing in speech:</strong> ${point}`;
            discrepancyList.appendChild(li);
        });

        // Add extra keypoints
        results.added_keypoints.forEach(point => {
            const li = document.createElement('li');
            li.innerHTML = `🔺 <strong>Extra in speech:</strong> ${point}`;
            discrepancyList.appendChild(li);
        });

        console.log("Updated discrepancies");
    }

    // Update title recommendations
    const titleRecommendationsSection = document.getElementById('titleRecommendationsSection');
    if (titleRecommendationsSection && results.suggested_titles && results.suggested_titles.length > 0) {
        titleRecommendationsSection.innerHTML = '<h5>Title Recommendations:</h5>';

        // Different reasons for each title
        const reasons = [
            "This title highlights the main focus and technical aspects of your research.",
            "This title emphasizes the innovative approach and potential impact of your work.",
            "This title presents your research in a professional academic context with clear focus.",
            "This title showcases the problem-solving aspects and practical applications of your work.",
            "This title emphasizes the methodological contributions of your research."
        ];

        // Add titles to the section
        results.suggested_titles.forEach((title, index) => {
            const div = document.createElement('div');
            div.className = 'recommendation-item';
            div.innerHTML = `
                <div class="recommendation-text">${title}</div>
                <div class="recommendation-reason">
                    ${reasons[index % reasons.length]}
                </div>
            `;
            titleRecommendationsSection.appendChild(div);
        });

        console.log(`Added ${results.suggested_titles.length} title recommendations`);
    }

    console.log("Analysis display updated successfully");
};

// Global function to generate titles
window.generateTitles = async function() {
    console.log("⭐⭐⭐ GLOBAL generateTitles FUNCTION CALLED ⭐⭐⭐");

    try {
        // Get the extracted text
        const extractedText = document.querySelector('#extracted-text')?.innerText || "";
        console.log(`Extracted text length: ${extractedText.length}`);

        // Get the speech text
        const speechText = window.transcriptText || "";
        console.log(`Speech text length: ${speechText.length}`);

        // Show loading indicator
        const titleRecommendationsSection = document.getElementById('titleRecommendationsSection');
        if (titleRecommendationsSection) {
            titleRecommendationsSection.innerHTML = `
                <h5 style="color: #6a1b9a; font-weight: bold; font-family: 'Comfortaa', sans-serif;">
                    Title Recommendations:
                </h5>
                <div style="
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    background: linear-gradient(135deg, #a18cd1 0%, #fbc2eb 100%);
                    padding: 20px;
                    border-radius: 16px;
                    box-shadow: 0 4px 20px rgba(128, 90, 213, 0.2);
                    margin-top: 10px;
                    font-family: 'Comfortaa', sans-serif;
                    color: #4a0072;
                ">
                    <div style="margin-right: 10px;">
                        <i class="fas fa-spinner fa-spin" style="font-size: 18px; color: #6a1b9a;"></i>
                    </div>
                    <div style="font-size: 15px; font-family: 'Comfortaa', sans-serif;">
                        Generating Title Recommendations...
                    </div>
                </div>
            `;
        }

        // Make the API call
        console.log("Making API call to /generate_title");
        const response = await fetch('/generate_title', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                extracted_text: extractedText,
                speech_text: speechText
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        // Parse the response
        const data = await response.json();
        console.log("Title generation results:", data);

        // Update the display
        if (titleRecommendationsSection && data.titles && data.titles.length > 0) {
            titleRecommendationsSection.innerHTML = '<h5>Title Recommendations:</h5>';

            // Different reasons for each title
            const reasons = [
                "This title highlights the main focus and technical aspects of your research.",
                "This title emphasizes the innovative approach and potential impact of your work.",
                "This title presents your research in a professional academic context with clear focus.",
                "This title showcases the problem-solving aspects and practical applications of your work.",
                "This title emphasizes the methodological contributions of your research."
            ];

            // Add titles to the section
            data.titles.forEach((title, index) => {
                const div = document.createElement('div');
                div.className = 'recommendation-item';
                div.innerHTML = `
                    <div class="recommendation-text">${title}</div>
                    <div class="recommendation-reason">
                        ${reasons[index % reasons.length]}
                    </div>
                `;
                titleRecommendationsSection.appendChild(div);
            });

            console.log(`Added ${data.titles.length} title recommendations`);
        } else {
            if (titleRecommendationsSection) {
                titleRecommendationsSection.innerHTML = `
                    <h5>Title Recommendations:</h5>
                    <div style="padding: 10px; color: #666;">
                        No title recommendations could be generated. Please try again.
                    </div>
                `;
            }
        }
        return data;
    } catch (error) {
        console.error("Error in generateTitles:", error);
        const titleRecommendationsSection = document.getElementById('titleRecommendationsSection');
        if (titleRecommendationsSection) {
            titleRecommendationsSection.innerHTML = `
                <h5>Title Recommendations:</h5>
                <div style="padding: 10px; color: #666;">
                    An error occurred while generating title recommendations. Please try again.
                </div>
            `;
        }
    }
};

// Global function to refresh the analysis
window.reAnalyzeAndShow = async function() {
    console.log("⭐⭐⭐ GLOBAL reAnalyzeAndShow FUNCTION CALLED ⭐⭐⭐");

    // Show loading state
    const button = document.querySelector('.refresh-button');
    if (button) {
        button.classList.add('loading');
    }

    const cornerLoading = document.querySelector('.corner-loading');
    if (cornerLoading) {
        cornerLoading.style.display = 'flex';
    }

    try {
        // Call the analysis function
        await window.analyzePDFContent();

        // Generate titles
        await window.generateTitles();

        console.log("Analysis refreshed successfully");
    } catch (error) {
        console.error("Error in reAnalyzeAndShow:", error);
        alert("An error occurred during analysis. Please try again.");
    } finally {
        // Hide loading state
        if (button) {
            button.classList.remove('loading');
        }

        if (cornerLoading) {
            cornerLoading.style.display = 'none';
        }
    }
};

// Initialize when the page loads
document.addEventListener('DOMContentLoaded', function() {
    console.log("⭐⭐⭐ GLOBAL ANALYSIS.JS LOADED ⭐⭐⭐");

    // Set up global variables
    window.transcriptText = window.transcriptText || "";

    // Set up arrow button handler with enhanced functionality
    const resultsButton = document.getElementById('resultsButton');
    if (resultsButton) {
        // Make the button more noticeable
        resultsButton.style.transition = 'all 0.3s ease';
        resultsButton.style.boxShadow = '0 4px 8px rgba(0, 0, 0, 0.2)';

        // Add a pulsing effect to draw attention
        const pulseAnimation = `
            @keyframes pulse {
                0% { transform: scale(1); box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); }
                50% { transform: scale(1.05); box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3); }
                100% { transform: scale(1); box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); }
            }
        `;

        const styleElement = document.createElement('style');
        styleElement.textContent = pulseAnimation;
        document.head.appendChild(styleElement);

        resultsButton.style.animation = 'pulse 2s infinite';

        // Enhanced click handler
        resultsButton.onclick = function() {
            console.log("Arrow button clicked - starting analysis");

            // Visual feedback
            resultsButton.style.animation = 'none';
            resultsButton.style.transform = 'scale(0.95)';
            setTimeout(() => {
                resultsButton.style.transform = 'scale(1)';
            }, 200);

            // Show the popup
            const popup = document.getElementById('popup');
            if (popup) {
                popup.style.display = 'flex';
            }

            // Run the analysis
            window.analyzePDFContent();

            return false;
        };

        // Add hover effect
        resultsButton.onmouseover = function() {
            resultsButton.style.transform = 'scale(1.1)';
        };

        resultsButton.onmouseout = function() {
            resultsButton.style.transform = 'scale(1)';
        };
    }

    const closeButton = document.getElementById('closeButton');
    if (closeButton) {
        closeButton.onclick = function() {
            console.log("Close button clicked");
            const popup = document.getElementById('popup');
            if (popup) {
                popup.style.display = 'none';
            }
            return false;
        };
    }

    const refreshButton = document.getElementById('refreshButton');
    if (refreshButton) {
        refreshButton.onclick = function() {
            console.log("Refresh button clicked");
            window.reAnalyzeAndShow();
            return false;
        };
    }

    // Load initial analysis data if available
    const analysisScript = document.getElementById('analysis-json');
    if (analysisScript && analysisScript.textContent && analysisScript.textContent.trim() !== '{}') {
        try {
            const analysisData = JSON.parse(analysisScript.textContent);
            if (analysisData) {
                window.updateAnalysisDisplay(analysisData);
            }
        } catch (e) {
            console.error("Failed to parse analysis data:", e);
        }
    }
});
