# UI Structure and UI/UX Design

## 1. UI Goal

Keep the interface direct and responsive so the user can draw a digit, inspect the result, and immediately try another one without leaving the page.

## 2. Page Layout

### Main Screen

The page should be organized around two primary zones:

- Drawing zone for canvas input and controls
- Results zone for prediction, confidence, probability breakdown, and recent history

### Structure

- Header with app title and short status text
- Drawing zone
- Results zone

## 3. Drawing Zone

The drawing zone should include:

- Main canvas with clear visual emphasis
- Clear, undo, redo, and eraser controls
- Brush size control for quick adjustments
- Touch and pointer support

## 4. Results Zone

The results zone should include:

- Predicted digit
- Confidence score
- Probability breakdown for all digits
- Short status message for prediction or rejection
- Recent prediction history
- Region switching when multiple digits are detected

## 5. Behavior

### Canvas

- Update predictions after a short debounce or stroke end rather than on every pointer event.
- Keep the drawing surface responsive on both desktop and mobile.
- Preserve the current drawing while the backend is processing.

### Output

- Show the top prediction prominently.
- Surface uncertainty honestly when the rejector flags a mark as ambiguous or not a digit.
- Keep the result panel stable until a new prediction arrives.

## 6. Interaction Flow

1. The user opens the app.
2. The user draws one or more digits.
3. The backend preprocesses the image and returns region-level predictions.
4. The UI updates the active result, probability breakdown, and history.
5. The user clears the canvas and repeats.

## 7. Visual Rules

- Give the canvas the most visual weight.
- Keep controls close to the canvas.
- Use clear contrast for prediction, confidence, and status.
- Make the page feel deliberate rather than generic.

## 8. Responsive Behavior

- Use a two-column layout on wider screens.
- Stack the zones on narrow screens.
- Keep the canvas large enough to draw comfortably at all sizes.

## 9. Summary

The UI should feel like a live digit recognition tool: draw, inspect, adjust, and repeat with minimal friction.
