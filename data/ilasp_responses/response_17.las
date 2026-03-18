%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Pre-processing                          : 0.008s
%% Hypothesis Space Generation             : 2.439s
%% Conflict analysis                       : 26.266s
%%   - Positive Examples                   : 26.266s
%% Counterexample search                   : 0.002s
%%   - CDOEs                               : 0s
%%   - CDPIs                               : 0.002s
%% Hypothesis Search                       : 8.543s
%% Total                                   : 38.5s
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d3(V1) :- d2(V1).
d4(V1) :- d3(V1).
d7(V1) :- d4(V1).
d1(V1) :- d4(V1).
d8(V1) :- d5(V1).
d0(V1) :- d5(V1).

