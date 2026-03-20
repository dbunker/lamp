%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Pre-processing                          : 0.008s
%% Hypothesis Space Generation             : 0.16s
%% Conflict analysis                       : 0.36s
%%   - Positive Examples                   : 0.36s
%% Counterexample search                   : 0.002s
%%   - CDOEs                               : 0s
%%   - CDPIs                               : 0s
%% Hypothesis Search                       : 0.113s
%% Total                                   : 0.667s
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d2(V1) :- d4(V1).
d1(V1) :- d0(V1).
d0(V1) :- d4(V1).

