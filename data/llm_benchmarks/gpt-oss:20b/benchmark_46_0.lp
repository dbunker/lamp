d5(o7).
d2(o5).
d1(o8).
d4(o9).
d4(o4).
d0(o8).
d2(o8).
d3(o8).
d1(o7).

d4(X) :- d2(X), not d3(X), not d5(X).
d4(X) :- d5(X).
d1(X) :- d2(X).