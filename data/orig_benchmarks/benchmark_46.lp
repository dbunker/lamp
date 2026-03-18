d5(o7).
d2(o5).
d1(o8).
d4(o9).
d4(o4).
d0(o8).
d2(o8).
d3(o8).
d1(o7).

d1(X) :- d2(X), not d5(X).
d4(X) :- not d3(X), d1(X).
d4(X) :- not d3(X), d1(X).
d5(X) :- d3(X), not d0(X).
d0(X) :- d5(X), not d1(X).
d5(X) :- not d2(X), d3(X).
d1(X) :- not d2(X), d3(X).
d1(X) :- d5(X), not d4(X).
d4(X) :- not d2(X), d1(X).
d4(X) :- d2(X), not d0(X).
d4(X) :- not d2(X), d3(X).