import { GoogleGenAI, Type } from "@google/genai";
import type { Vitals, EmotionalState } from '../src/types';

const API_KEY = process.env.API_KEY;

if (!API_KEY) {
  // This is a fallback for development. In a real environment, the key should be set.
  console.warn("Gemini API key not found. Using placeholder data. Set process.env.API_KEY.");
}

const ai = new GoogleGenAI({ apiKey: API_KEY! });

const getMockSuggestions = (emotion: EmotionalState) => {
    switch(emotion) {
        case 'Stressed': return ['Practice deep breathing for 2 minutes.', 'Listen to calming ambient sounds.', 'Take a short break to stretch.'];
        case 'Anxious': return ['Focus on a single, simple task.', 'Talk through your feelings with a crewmate or MAITRI.', 'Review mission protocols to reaffirm confidence.'];
        case 'Fatigued': return ['Ensure proper hydration.', 'Perform a 5-minute light exercise routine.', 'Confirm your next sleep cycle schedule.'];
        default: return ['Maintain current routine.', 'Engage in a recreational activity.', 'Check in with a fellow crew member.'];
    }
}

export const getInterventionSuggestions = async (vitals: Vitals | undefined, emotion: EmotionalState): Promise<string[]> => {
  if (!API_KEY) {
    return new Promise(resolve => setTimeout(() => resolve(getMockSuggestions(emotion)), 1000));
  }
  
  const prompt = `
    As an AI therapist for an astronaut in space, analyze the following data and provide three brief, actionable, and encouraging intervention suggestions. The astronaut's current emotional state is "${emotion}". Their physiological vitals are: Heart Rate ${vitals?.heartRate} BPM, Fatigue Index ${vitals?.fatigueIndex?.toFixed(2)}, and Posture is "${vitals?.posture}".
  `;
  
  try {
    const response = await ai.models.generateContent({
      model: "gemini-2.5-flash",
      contents: prompt,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            suggestions: {
              type: Type.ARRAY,
              description: "A list of three brief intervention suggestions.",
              items: { type: Type.STRING }
            }
          }
        },
        temperature: 0.7,
      },
    });

    const jsonString = response.text;
    const result = JSON.parse(jsonString);
    return result.suggestions || [];
  } catch (error) {
    console.error("Error fetching suggestions from Gemini:", error);
    // Fallback to mock data on API error
    return getMockSuggestions(emotion);
  }
};
