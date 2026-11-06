// JChronologer's own library predictions for requests read from stdin.
//
// Each line: peptide (JChronologer's UNIMOD form), TAB, charges (a comma
// list, or "auto:P" for the jar's automatic charge selection keeping charges
// with probability >= P), TAB, NCE. Each line is one
// LibraryPredictionRequest to ChronologerLibraryPredictor.predict, on its own,
// so a peptide the jar refuses does not stop the others (jchronologer
// 1.0.10, ChronologerFactory.createLibraryPredictor, CCS on, one inference
// thread, default batch size and minimum reported intensity).
//
// Output, tab-separated, one line per library entry: input, charges field,
// NCE, normalized UNIMOD peptide, precursor charge, precursor m/z, RT seconds,
// CCS (or NA), charge probability (or NA), ions ("label=mz=intensity" joined
// by ';', in the jar's order, Double/Float toString). A request the jar
// refuses prints input, charges, NCE, ERROR, message.
//
// Run by scripts/survey-jchronologer-predictions.py.

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

import org.searlelab.jchronologer.api.ChronologerLibraryEntry;
import org.searlelab.jchronologer.api.ChronologerLibraryOptions;
import org.searlelab.jchronologer.api.ChronologerLibraryPredictor;
import org.searlelab.jchronologer.api.LibraryPredictionRequest;
import org.searlelab.jchronologer.api.PrecursorCondition;
import org.searlelab.jchronologer.impl.ChronologerFactory;

public class PredictionOracle {
    public static void main(String[] args) throws Exception {
        ChronologerLibraryOptions options = ChronologerLibraryOptions.builder()
                .ccsPredictionEnabled(true)
                .inferenceThreads(1)
                .build();
        ChronologerLibraryPredictor predictor = ChronologerFactory.createLibraryPredictor(options);
        predictor.init();
        BufferedReader in = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));
        String line;
        while ((line = in.readLine()) != null) {
            if (line.isBlank()) {
                continue;
            }
            String[] f = line.split("\t");
            String peptide = f[0];
            String charges = f[1];
            double nce = Double.parseDouble(f[2]);
            LibraryPredictionRequest request;
            if (charges.startsWith("auto:")) {
                request = new LibraryPredictionRequest(peptide, nce, Double.parseDouble(charges.substring(5)));
            } else {
                List<PrecursorCondition> conditions = new ArrayList<>();
                for (String z : charges.split(",")) {
                    conditions.add(new PrecursorCondition(Byte.parseByte(z), nce));
                }
                request = new LibraryPredictionRequest(peptide, conditions);
            }
            String head = peptide + "\t" + charges + "\t" + f[2];
            List<ChronologerLibraryEntry> entries;
            try {
                entries = predictor.predict(List.of(request));
            } catch (RuntimeException e) {
                System.out.println(head + "\tERROR\t" + String.valueOf(e.getMessage()).replace('\t', ' ').replace('\n', ' '));
                continue;
            }
            for (ChronologerLibraryEntry entry : entries) {
                StringBuilder ions = new StringBuilder();
                double[] mz = entry.getMassArray();
                float[] intensity = entry.getIntensityArray();
                String[] labels = entry.getIonTypeArray();
                for (int i = 0; i < mz.length; i++) {
                    if (i > 0) {
                        ions.append(';');
                    }
                    ions.append(labels[i]).append('=').append(mz[i]).append('=').append(intensity[i]);
                }
                System.out.println(head
                        + "\t" + entry.getUnimodPeptideSequence()
                        + "\t" + entry.getPrecursorCharge()
                        + "\t" + entry.getPrecursorMz()
                        + "\t" + entry.getRetentionTimeInSeconds()
                        + "\t" + entry.getCCS().map(String::valueOf).orElse("NA")
                        + "\t" + entry.getChargeProbability().map(String::valueOf).orElse("NA")
                        + "\t" + ions);
            }
        }
        predictor.close();
    }
}
