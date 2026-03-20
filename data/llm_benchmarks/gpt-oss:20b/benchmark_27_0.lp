d2(o3).
d7(o0).
d6(o1).
d4(o5).
d5(o0).
d7(o2).
d9(o2).
d6(o4).
d3(o2).

d5(o2) :- d7(o2), d5(o0).
d5(o5) :- d6(o5), d5(o0).
d6(o0) :- d5(o0), d0(o0).
d6(o5) :- d5(o0), d0(o5).
d8(O) :- not d0(O), d9(O).
d1(O) :- d7(O), not d0(O).
d0(O) :- d9(O), not d1(O).
d0(O) :- d8(O), not d1(O).
d0(O) :- d7(O), not d1(O).
d0(O) :- not d1(O), d6(O).
d0(O) :- d5(O), not d1(O).
d0(O) :- d4(O), not d1(O).
d0(O) :- d3(O), not d1(O).
d0(O) :- d2(O), not d1(O).