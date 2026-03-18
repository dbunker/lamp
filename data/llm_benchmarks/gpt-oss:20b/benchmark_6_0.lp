d0(o7).
d1(o4).
d8(o6).
d5(o5).
d2(o0).
d9(o2).
d7(o6).
d3(o8).
d1(o3).

d2(X) :- d3(X).
d4(X) :- not d5(X), d6(X).
d3(X) :- not d5(X), d6(X).
d6(X) :- d7(X).
d7(X) :- d1(X).
d5(X) :- d1(X).