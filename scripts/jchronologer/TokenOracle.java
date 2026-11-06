// The tokens JChronologer itself gives Chronologer, Cartographer and Sculptor
// for each peptide read from stdin, one per line in JChronologer's UNIMOD form
// ("[UNIMOD:1]-PEPTM[UNIMOD:35]IDEK-[]").
//
// Each model's path is the one DefaultChronologerLibraryPredictor.predict takes
// (jchronologer 1.0.10): normalizeToUnimod; Chronologer preprocesses the
// mass-encoded form (DefaultChronologer.predict); Cartographer the mass
// spelling rebuilt after foldFirstResiduePyrogluToNterm (predict, the loop
// over massByUnimod, copied below); Sculptor tokenizes the parsed UNIMOD form
// (inferCCSByJob, through the private SculptorTokenizer).
//
// Output, tab-separated: input, model, accepted|refused, detail, tokens
// (space-separated ids, padding included; empty when refused).
//
// Run by scripts/survey-jchronologer-tokens.py.

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.lang.reflect.Constructor;
import java.lang.reflect.Method;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import org.searlelab.jchronologer.preprocessing.ChronologerPreprocessor;
import org.searlelab.jchronologer.preprocessing.PeptideSequenceConverter;
import org.searlelab.jchronologer.preprocessing.PeptideSequenceConverter.ParsedUnimodSequence;
import org.searlelab.jchronologer.preprocessing.PreprocessingMetadataLoader;
import org.searlelab.jchronologer.preprocessing.PreprocessingOutcome;

public class TokenOracle {
    private static final double EPSILON = 1e-5; // ChronologerLibraryOptions.DEFAULT_MASS_MATCH_EPSILON

    public static void main(String[] args) throws Exception {
        ChronologerPreprocessor chronologer =
                new ChronologerPreprocessor(PreprocessingMetadataLoader.loadFromClasspath(args[0]));
        ChronologerPreprocessor cartographer =
                new ChronologerPreprocessor(PreprocessingMetadataLoader.loadFromClasspath(args[1]));

        Class<?> library = Class.forName("org.searlelab.jchronologer.impl.DefaultChronologerLibraryPredictor");
        Method loadSculptor = library.getDeclaredMethod("loadSculptorMetadata", String.class);
        loadSculptor.setAccessible(true);
        Object metadata = loadSculptor.invoke(null, args[2]);
        Class<?> tokenizerClass =
                Class.forName("org.searlelab.jchronologer.impl.DefaultChronologerLibraryPredictor$SculptorTokenizer");
        Constructor<?> constructor = tokenizerClass.getDeclaredConstructors()[0];
        constructor.setAccessible(true);
        Object sculptor = constructor.newInstance(metadata);
        Method tokenize = tokenizerClass.getDeclaredMethod("tokenize", ParsedUnimodSequence.class);
        tokenize.setAccessible(true);

        BufferedReader in = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));
        String line;
        while ((line = in.readLine()) != null) {
            if (line.isBlank()) {
                continue;
            }
            String unimod;
            try {
                unimod = PeptideSequenceConverter.normalizeToUnimod(line, EPSILON);
            } catch (RuntimeException e) {
                for (String model : new String[] {"chronologer", "cartographer", "sculptor"}) {
                    print(line, model, "refused", "normalize: " + e.getMessage(), null);
                }
                continue;
            }
            try {
                outcome(line, "chronologer", chronologer.preprocess(PeptideSequenceConverter.unimodToMassEncoded(unimod)));
            } catch (RuntimeException e) {
                print(line, "chronologer", "refused", "error: " + e.getMessage(), null);
            }
            try {
                outcome(line, "cartographer", cartographer.preprocess(cartographerMassEncoded(unimod)));
            } catch (RuntimeException e) {
                print(line, "cartographer", "refused", "error: " + e.getMessage(), null);
            }
            long[] tokens = (long[]) tokenize.invoke(sculptor, PeptideSequenceConverter.parseNormalizedUnimod(unimod));
            print(line, "sculptor", tokens == null ? "refused" : "accepted", tokens == null ? "null" : unimod, tokens);
        }
    }

    /** DefaultChronologerLibraryPredictor.predict, the Cartographer mass spelling. */
    private static String cartographerMassEncoded(String unimod) {
        ParsedUnimodSequence parsed = PeptideSequenceConverter.parseNormalizedUnimod(unimod);
        List<String> ntermMods = new ArrayList<>(parsed.getPositionMods().get(0));
        List<List<String>> residueMods = new ArrayList<>(parsed.getResidues().length());
        for (int ri = 0; ri < parsed.getResidues().length(); ri++) {
            residueMods.add(new ArrayList<>(parsed.getPositionMods().get(ri + 1)));
        }
        PeptideSequenceConverter.foldFirstResiduePyrogluToNterm(parsed.getResidues(), ntermMods, residueMods);
        StringBuilder out = new StringBuilder();
        if (!ntermMods.isEmpty()) {
            out.append('[')
                    .append(String.format(Locale.US, "%+.6f", PeptideSequenceConverter.sumUnimodMass(ntermMods)))
                    .append(']');
        }
        for (int ri = 0; ri < parsed.getResidues().length(); ri++) {
            out.append(parsed.getResidues().charAt(ri));
            List<String> mods = residueMods.get(ri);
            if (!mods.isEmpty()) {
                out.append('[')
                        .append(String.format(Locale.US, "%+.6f", PeptideSequenceConverter.sumUnimodMass(mods)))
                        .append(']');
            }
        }
        return out.toString();
    }

    private static void outcome(String input, String model, PreprocessingOutcome outcome) {
        if (outcome.isAccepted()) {
            print(input, model, "accepted", outcome.getCodedPeptideSeq(), outcome.getTokenArray());
        } else {
            String detail = String.valueOf(outcome.getRejectionReason());
            if (outcome.getErrorDetail() != null) {
                detail += ": " + outcome.getErrorDetail();
            }
            print(input, model, "refused", detail, null);
        }
    }

    private static void print(String input, String model, String status, String detail, long[] tokens) {
        StringBuilder ids = new StringBuilder();
        if (tokens != null) {
            for (int i = 0; i < tokens.length; i++) {
                if (i > 0) {
                    ids.append(' ');
                }
                ids.append(tokens[i]);
            }
        }
        System.out.println(input + "\t" + model + "\t" + status + "\t" + detail.replace('\t', ' ') + "\t" + ids);
    }
}
