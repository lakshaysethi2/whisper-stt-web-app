describe('File upload: keep file on model error', () => {
  it('shows model error but keeps file name visible', () => {
    cy.visit('/');

    // Manually trigger file selection by clicking the file drop zone,
    // then use selectFile on the file input
    cy.get('#file-input').invoke('removeAttr', 'hidden');
    cy.get('#file-input').selectFile('cypress/fixtures/test-audio.mp3');

    // File name should be visible
    cy.get('#file-name').should('be.visible');
    cy.get('#file-name').should('contain.text', 'test-audio.mp3');

    // Transcribe button should be visible
    cy.get('#transcribe-file-btn').should('be.visible');

    // Click Transcribe without model selected
    cy.get('#transcribe-file-btn').click();

    // Model error should appear
    cy.get('#model-error').should('be.visible');
    cy.get('#model-error').should('contain.text', 'Please select a model');

    // File name should still be visible (should NOT have been reset)
    cy.get('#file-name').should('be.visible');
    cy.get('#file-name').should('contain.text', 'test-audio.mp3');

    // File drop text should still say "File selected"
    cy.get('#file-drop-text').should('contain.text', 'File selected');

    // Select a model
    cy.get('#model-select').select('base');

    // Model error should clear
    cy.get('#model-error').should('not.be.visible');

    // File name should still be visible
    cy.get('#file-name').should('be.visible');
    cy.get('#file-name').should('contain.text', 'test-audio.mp3');
  });
});
