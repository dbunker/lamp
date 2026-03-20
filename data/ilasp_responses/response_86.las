%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Pre-processing                          : 0.008s
%% Hypothesis Space Generation             : 0.151s
%% Conflict analysis                       : 0.394s
%%   - Positive Examples                   : 0.394s
%% Counterexample search                   : 0.002s
%%   - CDOEs                               : 0s
%%   - CDPIs                               : 0s
%% Hypothesis Search                       : 0.145s
%% Total                                   : 0.721s
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d5(V1) :- d2(V1).
d5(V1) :- d0(V1).
d4(V1) :- d0(V1).
d3(V1) :- d4(V1); not d0(V1).

