d8(o6).
d3(o7).
d9(o9).
d8(o7).
d4(o4).
d9(o8).
d6(o4).
d1(o3).
d0(o6).

d7(X) :- d4(X).
d2(X) :- d4(X).
d5(X) :- d9(X), not d1(X).
d9(X) :- d3(X).
d9(X) :- d4(X).
d4(X) :- not d8(X), d3(X).
d3(X) :- d1(X).